#!/usr/bin/env python3
import argparse
import base64
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from yieldcurve.data import TARGETS, MATURITIES, load_yields, record_provenance
from yieldcurve.forecast import MODELS
from yieldcurve.evaluate import forecast_metrics, compare_to_paper, paired_comparisons


def table(frame, digits=3):
    return frame.to_html(index=False, border=0, float_format=lambda x: f"{x:.{digits}f}", classes="results", na_rep="—")


def embedded(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def make_figures(metrics, forecasts, importance, output, trees):
    figs = output / "figures"
    figs.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    order = [m for m in MODELS if m in metrics.model.unique()]
    horizons = sorted(metrics.horizon.unique())
    figure, axes = plt.subplots(2, 2, figsize=(22, 14), constrained_layout=True)
    normalizer = TwoSlopeNorm(vmin=.60, vcenter=1., vmax=2.50)
    for axis, horizon in zip(axes.ravel(), horizons):
        selected = metrics[metrics.horizon == horizon]
        values = selected.pivot(index="model", columns="target", values="relative_rmse").reindex(index=order, columns=TARGETS)
        im = axis.imshow(values, aspect="auto", cmap="RdBu_r", norm=normalizer)
        axis.set_xticks(range(len(TARGETS)), [t.replace("_", "–") for t in TARGETS], rotation=45)
        axis.set_yticks(range(len(order)), order)
        axis.set_title(f"{horizon}-month horizon · n={int(selected.n.min())}", fontweight="bold")
        for i in range(len(order)):
            for j in range(len(TARGETS)):
                val = values.iloc[i, j]
                if np.isfinite(val):
                    axis.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                              color="white" if val > 1.65 or val < .72 else "#15283f")
        axis.set_xlabel("Maturity or spread, months")
    for unused in axes.ravel()[len(horizons):]:
        unused.set_visible(False)
    figure.suptitle(f"Computed relative RMSE · {trees}-tree replication\nBelow 1 beats the random walk; PSO and feature definitions include disclosed assumptions", fontsize=16)
    figure.colorbar(im, ax=axes, shrink=.65, label="RMSE / random-walk RMSE", extend="both")
    figure.savefig(figs / "relative_rmse.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    first_horizon = min(horizons)
    for axis, target in zip(axes.ravel(), ["3", "24", "120", "24_120"]):
        group = forecasts[(forecasts.horizon == first_horizon) & (forecasts.target == target) &
                          (forecasts.split == "oos")]
        for model, color in [("Direct-RF-X", "#177f8e"), ("Random Walk", "#d39a36")]:
            sub = group[group.model == model].sort_values("target_month")
            axis.plot(pd.PeriodIndex(sub.target_month, freq="M").to_timestamp(), 100 * sub.predicted,
                      color=color, linewidth=1.1, label=model)
        sub = group[group.model == "Random Walk"].sort_values("target_month")
        axis.plot(pd.PeriodIndex(sub.target_month, freq="M").to_timestamp(), 100 * sub.realized,
                  color="#162b42", linewidth=1.2, alpha=.8, label="Realized")
        axis.set_title(f"{target.replace('_', '–')} months")
        axis.set_ylabel("Yield / spread (%)")
        axis.grid(alpha=.15)
    axes[0, 0].legend(frameon=False, fontsize=9)
    figure.suptitle(f"{first_horizon}-month forecasts · original zero-coupon Treasury data", fontsize=15)
    figure.savefig(figs / "forecast_paths.png", dpi=160)
    plt.close(figure)

    if not importance.empty and "Direct-RF-X" in set(importance.model):
        selected = importance[importance.model == "Direct-RF-X"].copy()
        selected["group"] = np.where(selected.feature.str.startswith("macro_"), "Macro", "Yield history")
        groups = selected.groupby(["horizon", "response", "group"]).importance.sum().unstack("group").groupby("response").mean().reindex(TARGETS)
        figure, axis = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
        x = np.arange(len(groups))
        axis.bar(x, 100 * groups["Macro"], color="#177f8e", label="Macro")
        axis.bar(x, 100 * groups["Yield history"], bottom=100 * groups["Macro"], color="#d5a247", label="Yield history")
        axis.set_xticks(x, [t.replace("_", "–") for t in TARGETS])
        axis.set_ylim(0, 100)
        axis.set_ylabel("Share of impurity importance (%)")
        axis.set_xlabel("Response maturity / spread, months")
        axis.set_title("Direct-RF-X feature importance · average across origins and horizons")
        axis.legend(frameon=False, loc="upper right")
        figure.savefig(figs / "macro_importance.png", dpi=160)
        plt.close(figure)
    return figs


def make_curve_figures(output, importance):
    """Recreate the main descriptive curve and calibration diagnostics."""
    figs = output / "figures"
    yields = load_yields(ROOT / "data/raw/yield_monthly.csv").loc["1986-05":"2025-06"]
    maturities = np.r_[np.arange(1, 360, 6), 360]
    z = 100 * yields[list(map(str, maturities))].to_numpy()
    months = np.arange(len(yields))
    x, y = np.meshgrid(months, maturities / 12)
    figure = plt.figure(figsize=(12, 7), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    surface = axis.plot_surface(x, y, z.T, cmap="viridis", rstride=2, cstride=5, linewidth=0, antialiased=True)
    ticks = [0, 120, 240, 360, len(yields) - 1]
    axis.set_xticks(ticks, [str(yields.index[i]) for i in ticks], rotation=10)
    axis.set_xlabel("Observation month", labelpad=12)
    axis.set_ylabel("Maturity (years)", labelpad=12)
    axis.set_zlabel("Zero-coupon yield (%)", labelpad=8)
    axis.set_title("Original Treasury zero-coupon yield surface · May 1986–June 2025", pad=20)
    axis.view_init(elev=25, azim=-130)
    figure.colorbar(surface, ax=axis, shrink=.6, pad=.10, label="Yield (%)")
    figure.savefig(figs / "yield_surface.png", dpi=160)
    plt.close(figure)
    pso_path = output / "factors_DNSS-PSO.csv"
    if pso_path.exists():
        from yieldcurve.curves import loadings
        pso = pd.read_csv(pso_path, index_col=0)
        pso.index = pd.PeriodIndex(pso.index, freq="M")
        pso = pso.reindex(yields.index)
        dates = pso.index.to_timestamp()
        figure, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
        for axis, column in zip(axes.ravel(), ["beta0", "beta1", "beta2", "beta3", "lambda1", "lambda2"]):
            multiplier = 100 if column.startswith("beta") else 1
            axis.plot(dates, multiplier * pso[column], color="#177f8e", linewidth=.9)
            axis.set_title(column)
            axis.set_ylabel("Coefficient (% yield)" if multiplier == 100 else "Monthly decay")
            axis.grid(alpha=.15)
        figure.suptitle("Time-varying PSO decays and conditional OLS factors\nIndependent optimizer; weak factor identification remains visible", fontsize=14)
        figure.savefig(figs / "pso_parameters.png", dpi=150)
        plt.close(figure)
        residuals = []
        for month, row in pso.iterrows():
            observed = yields.loc[month, list(map(str, MATURITIES))].to_numpy()
            predicted = loadings(MATURITIES, row.lambda1, row.lambda2) @ row[["beta0", "beta1", "beta2", "beta3"]].to_numpy()
            residuals.append(np.sqrt(np.mean((observed - predicted) ** 2)))
        worst = np.argsort(residuals)[-4:][::-1]
        figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        grid = np.linspace(1, 360, 500)
        for axis, position in zip(axes.ravel(), worst):
            month = pso.index[position]
            row = pso.iloc[position]
            predicted = loadings(grid, row.lambda1, row.lambda2) @ row[["beta0", "beta1", "beta2", "beta3"]].to_numpy()
            axis.plot(grid / 12, 100 * predicted, color="#177f8e", label="PSO/OLS fitted curve")
            axis.scatter(np.array(MATURITIES) / 12, 100 * yields.loc[month, list(map(str, MATURITIES))],
                         color="#162b42", s=25, label="Observed fitting maturities")
            axis.set_title(f"{month} · RMSE {100 * residuals[position]:.3f} pp")
            axis.set_xlabel("Maturity (years)")
            axis.set_ylabel("Yield (%)")
            axis.grid(alpha=.15)
        axes[0, 0].legend(frameon=False, fontsize=9)
        figure.suptitle("Four largest cross-sectional fit errors in this replication", fontsize=14)
        figure.savefig(figs / "worst_curve_fits.png", dpi=160)
        plt.close(figure)
    if not importance.empty and "Direct-RF-X" in set(importance.model):
        meta = pd.read_csv(ROOT / "references/macro_variables.csv").set_index("code")
        selected = importance[(importance.model == "Direct-RF-X") & importance.feature.str.startswith("macro_")].copy()
        selected["code"] = selected.feature.str.slice(6).str.replace(r"_lag[456]$", "", regex=True)
        selected["category"] = selected.code.map(meta.category)
        groups = selected.groupby(["horizon", "response", "category"]).importance.sum().unstack("category").fillna(0)
        groups = groups.div(groups.sum(axis=1), axis=0).groupby("response").mean().reindex(TARGETS)
        groups.to_csv(output / "macro_category_importance.csv")
        selected.groupby(["horizon", "response", "code"]).importance.sum().groupby(["response", "code"]).mean().reset_index().sort_values(["response", "importance"], ascending=[True, False]).to_csv(output / "macro_variable_importance.csv", index=False)
        colors = plt.get_cmap("tab10")(np.linspace(0, .9, len(groups.columns)))
        figure, axis = plt.subplots(figsize=(13, 6), constrained_layout=True)
        bottom = np.zeros(len(groups))
        for category, color in zip(groups.columns, colors):
            vals = 100 * groups[category].to_numpy()
            axis.bar(np.arange(len(groups)), vals, bottom=bottom, color=color, label=category)
            bottom += vals
        axis.set_xticks(np.arange(len(groups)), [t.replace("_", "–") for t in TARGETS])
        axis.set_ylim(0, 100)
        axis.set_ylabel("Share within the macro block (%)")
        axis.set_xlabel("Response maturity / spread, months")
        axis.set_title("Composition of macro importance · Direct-RF-X")
        axis.legend(frameon=False, bbox_to_anchor=(.5, -.18), loc="upper center", ncol=3, fontsize=9)
        figure.savefig(figs / "macro_categories.png", dpi=160)
        plt.close(figure)
    trade_path = output / "trades.csv"
    if trade_path.exists():
        trades = pd.read_csv(trade_path)
        group = trades[(trades.horizon == 1) & (trades.target == "24_120")]
        figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
        for model in ["Direct-RF-X", "Direct-RF", "DNS-AR", "DNSS-AR", "Roll-down"]:
            sub = group[group.model == model].sort_values("origin")
            if not sub.empty:
                dates = (pd.PeriodIndex(sub.origin, freq="M") + 1).to_timestamp()
                axis.plot(dates, (1 + sub.total_net).cumprod(), label=model, linewidth=1.2)
        axis.set_title("One-month 2y–10y slope trades · independent synthetic-bond simulation")
        axis.set_ylabel("Compounded net return index (initial value = 1)")
        axis.grid(alpha=.15)
        axis.legend(frameon=False, ncol=3, fontsize=9)
        figure.savefig(figs / "one_month_slope_trades.png", dpi=160)
        plt.close(figure)


def notebook(output):
    def md(source):
        return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}
    def code(source):
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": source.splitlines(keepends=True)}
    cells = [
        md("# Yield-curve paper reproduction\n\nIndependent implementation of [arXiv:2608.07536v1](https://arxiv.org/html/2608.07536v1).\n\nTree count and model scope are recorded in the manifest below. The supplied replication uses 50 trees versus the paper's 500. Exact author-code replication is limited by missing specifications; see `references/REPRODUCTION_DECISIONS.md`."),
        code("from pathlib import Path\nimport sys, json\nimport pandas as pd\nimport matplotlib.pyplot as plt\nROOT = Path.cwd()\nif not (ROOT / 'src').exists():\n    raise RuntimeError('Run this notebook from the extracted project folder.')\nsys.path.insert(0, str(ROOT / 'src'))\nRESULTS = ROOT / " + repr(str(output.relative_to(ROOT))) + "\nmanifest = json.loads((RESULTS / 'manifest.json').read_text())\nmanifest['config']"),
        md("## Input and benchmark fidelity\n\nThe original data reproduce the Table 1 statistics. The recovered origin schedule and AR indexing match the benchmark results at displayed precision."),
        code("summary = pd.read_csv(RESULTS / 'summary_statistics_comparison.csv')\ncomparison = pd.read_csv(RESULTS / 'paper_comparison.csv', dtype={'target': str})\ncomparison.groupby(['model', 'metric']).agg(\n    rounded_matches=('rounds_to_paper', 'sum'),\n    cells=('rounds_to_paper', 'size'),\n    mean_absolute_difference=('difference', lambda s: s.abs().mean())\n)"),
        md("## Forecast results\n\nRMSE is in percentage points. Relative RMSE below one beats the random walk. These are recomputed values, separate from the paper's reference tables."),
        code("metrics = pd.read_csv(RESULTS / 'forecast_metrics.csv', dtype={'target': str})\nmetrics.query('horizon == 1').pivot(index='model', columns='target', values='relative_rmse').round(3)"),
        code("from IPython.display import Image, display\ndisplay(Image(filename=str(RESULTS / 'figures/relative_rmse.png')))"),
        md("## Compare macro augmentation\n\nPaired DM tests use aligned dates and HAC uncertainty. A publication lag is implemented; revised macro snapshots are not real-time data vintages."),
        code("paired = pd.read_csv(RESULTS / 'paired_model_comparisons.csv', dtype={'target': str})\npaired[(paired['model'] == 'Direct-RF-X') & (paired['benchmark'] == 'Direct-RF')].sort_values(['horizon','target'])"),
        code("display(Image(filename=str(RESULTS / 'figures/macro_importance.png')))"),
        md("## Curve fitting and trading\n\nTrading uses synthetic aged bonds, gross-normalized DV01-neutral slope positions, and overlapping holding periods. It is an independent backtest reconstruction; its returns do not reproduce the paper's undisclosed normalization exactly."),
        code("pd.read_csv(RESULTS / 'curve_fit_comparison.csv')"),
        code("pd.read_csv(RESULTS / 'trading_summary.csv').sort_values(['trade_type','average_rank_total'])"),
        md("## Rerun\n\nUse the terminal in this project folder:\n\n```bash\npython -m pip install -r requirements.txt\npython run.py --config configs/full_500_trees.json --output results/full_500 --workers 8\npython make_report.py --results results/full_500\n```\n\nThe full run can take several hours. Optional recursive validation is in `tune.py`. For transformations matching official FRED-MD rather than Appendix A, use `configs/official_fred_sensitivity.json`."),
    ]
    doc = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
           "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, cell in enumerate(cells):
        cell["id"] = f"yieldcurve-{i:02d}"
    (ROOT / "reproduction.ipynb").write_text(json.dumps(doc, indent=1) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build a report from the actual reproduction results")
    parser.add_argument("--results", default="results/replication_50trees")
    args = parser.parse_args()
    output = ROOT / args.results
    forecasts = pd.read_csv(output / "forecasts.csv", dtype={"target": str})
    manifest = json.loads((output / "manifest.json").read_text())
    config = manifest["config"]
    # Recompute derived statistics with the final evaluator, without refitting forecasts.
    metrics = forecast_metrics(forecasts)
    metrics.to_csv(output / "forecast_metrics.csv", index=False)
    reference = pd.read_csv(ROOT / "references/paper_forecast_metrics.csv", dtype={"target": str})
    comparison = compare_to_paper(metrics, reference)
    comparison.to_csv(output / "paper_comparison.csv", index=False)
    paired = paired_comparisons(forecasts)
    paired.to_csv(output / "paired_model_comparisons.csv", index=False)
    importance_path = output / "feature_importance.csv"
    importance = pd.read_csv(importance_path, dtype={"response": str}) if importance_path.exists() else pd.DataFrame()
    figs = make_figures(metrics, forecasts, importance, output, config["forest_trees"])
    make_curve_figures(output, importance)
    summary = pd.read_csv(output / "summary_statistics_comparison.csv")
    checks = [
        {"Check": "Table 1 yield summary statistics", "Matches": int(summary.rounds_to_paper.sum()), "Cells": len(summary), "Precision": "3 decimals"},
    ]
    for model, metric, label in [("Random Walk", "rmse_pp", "Random-walk absolute RMSE"),
                                 ("DNS-AR", "relative_rmse", "DNS-AR relative RMSE"),
                                 ("DNSS-AR", "relative_rmse", "DNSS-AR relative RMSE"),
                                 ("DNS-AR", "rmse_pp", "DNS-AR absolute RMSE"),
                                 ("DNSS-AR", "rmse_pp", "DNSS-AR absolute RMSE")]:
        subset = comparison[(comparison.model == model) & (comparison.metric == metric)]
        checks.append({"Check": label, "Matches": int(subset.rounds_to_paper.sum()), "Cells": len(subset), "Precision": "2 decimals"})
    fidelity = pd.DataFrame(checks)
    fidelity.to_csv(output / "fidelity_checks.csv", index=False)
    averaging = metrics[metrics.model != "Random Walk"].copy()
    averaging["segment"] = averaging.target.map(lambda x: "Slope" if "_" in x else "Short end" if int(x) <= 24 else "Belly" if int(x) <= 60 else "Long end")
    aggregate = averaging.pivot_table(index="model", columns="segment", values="relative_rmse", aggfunc="mean")
    aggregate = aggregate.reindex(columns=["Short end", "Belly", "Long end", "Slope"]).sort_values("Slope").reset_index()
    aggregate.to_csv(output / "segment_average_relative_rmse.csv", index=False)
    curve_fit = pd.read_csv(output / "curve_fit_comparison.csv")
    scaled_path = output / "optimizer_scaled_sensitivity_forecast_metrics.csv"
    optimizer_sensitivity = pd.DataFrame()
    if scaled_path.exists():
        scaled = pd.read_csv(scaled_path, dtype={"target": str})
        optimizer_sensitivity = metrics[metrics.model.str.contains("-PSO")].merge(
            scaled[["model", "horizon", "target", "relative_rmse"]],
            on=["model", "horizon", "target"], suffixes=("_unscaled", "_scaled"))
        optimizer_sensitivity.to_csv(output / "optimizer_sensitivity_comparison.csv", index=False)
        optimizer_sensitivity = optimizer_sensitivity.groupby(["model", "horizon"])[
            ["relative_rmse_unscaled", "relative_rmse_scaled"]].mean().reset_index()
    trade_summary_path = output / "trading_summary.csv"
    trading = pd.read_csv(trade_summary_path) if trade_summary_path.exists() else pd.DataFrame()
    direct_x = paired[(paired.model == "Direct-RF-X") & (paired.benchmark == "Direct-RF")]
    slope_pairs = direct_x[direct_x.target.str.contains("_")]
    lower_count = int((slope_pairs.relative_rmse < 1).sum())
    finding = f"Direct-RF-X has lower RMSE than Direct-RF in {lower_count}/{len(slope_pairs)} slope/horizon comparisons in this run."
    observations = metrics[metrics.model == "Random Walk"].groupby("horizon").n.first().to_dict()
    counts = f"{len(config['models'])} competing models + random walk · {len(config['horizons'])} horizons · 13 targets · {len(metrics):,} metric rows"
    brief = {
        "status": "completed_reduced_tree_replication" if config["forest_trees"] != 500 else "completed_500_tree_independent_replication",
        "forest_trees": config["forest_trees"], "paper_forest_trees": 500,
        "models": len(config["models"]), "forecast_metric_rows": len(metrics),
        "forecast_rows": len(forecasts), "oos_observations_by_horizon": observations,
        "fidelity_checks": checks, "macro_slope_comparisons_improved": lower_count,
        "macro_slope_comparisons": len(slope_pairs),
        "exact_ml_replication_claimed": False, "exact_trading_replication_claimed": False,
        "time_varying_pso_forecasts_replicated": False,
    }
    (output / "run_summary.json").write_text(json.dumps(brief, indent=2) + "\n")
    # All final derived quantities now correspond to the current evaluator.
    updated = record_provenance(ROOT, config)
    updated["package_versions"] = {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib")}
    source_files = list(ROOT.glob("*.py")) + list((ROOT / "src").rglob("*.py")) + list((ROOT / "tests").glob("*.py"))
    updated["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(source_files)}
    updated["fingerprint"] = hashlib.sha256(json.dumps(updated, sort_keys=True).encode()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(updated, indent=2) + "\n")
    images = [("Relative forecast error", figs / "relative_rmse.png"), ("Forecast paths", figs / "forecast_paths.png"),
              ("Yield-curve data", figs / "yield_surface.png")]
    if (figs / "macro_importance.png").exists():
        images.append(("Macro feature importance", figs / "macro_importance.png"))
    if (figs / "macro_categories.png").exists():
        images.append(("Macro categories", figs / "macro_categories.png"))
    panels_html = "".join(f'<section><h2>{title}</h2><img src="{embedded(path)}" alt="{title}"></section>' for title, path in images)
    title = "Yield-curve reproduction"
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>
<style>body{{margin:0;background:#f4f6f8;color:#173047;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1180px;margin:0 auto;padding:42px 30px 70px}}header{{border-bottom:3px solid #187f8c;padding-bottom:28px}}h1{{font-size:42px;line-height:1.1;margin:12px 0}}h2{{margin-top:0;font-size:24px}}h3{{font-size:19px}}.eyebrow{{text-transform:uppercase;letter-spacing:.12em;color:#187f8c;font-size:12px;font-weight:700}}.status{{display:inline-block;background:#e5f1f2;color:#145761;padding:5px 12px;border-radius:5px;font-size:13px;font-weight:700}}.notice{{background:#fff5df;border-left:4px solid #c39330;padding:14px 18px;margin-top:22px}}section{{background:white;padding:25px;margin-top:25px;border:1px solid #dce4e9;border-radius:8px;overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th{{text-align:left;border-bottom:2px solid #9bb1be;padding:9px;color:#173047}}td{{border-bottom:1px solid #e1e8ed;padding:9px}}tr:nth-child(even){{background:#f8fafb}}img{{max-width:100%;height:auto}}p{{margin:10px 0}}a{{color:#146f81}}code{{font-size:14px;background:#eef3f6;padding:2px 5px}}pre{{padding:18px;background:#173047;color:#fff;overflow:auto;border-radius:5px}}pre code{{background:none;color:inherit}}.muted{{color:#617788;font-size:14px}}footer{{margin-top:30px;color:#617788;font-size:13px}}</style></head>
<body><main><header><div class="eyebrow">Independent research reproduction · arXiv:2608.07536v1</div><h1>{title}</h1><p>Original Treasury zero-coupon data, macro predictors, and direct versus factor forecasts.</p><span class="status">Completed: {config['forest_trees']} trees per forest</span><p class="muted">{counts}</p><div class="notice"><strong>Scope:</strong> The paper uses 500 trees. The supplied run uses {config['forest_trees']}. Data and baseline AR forecasts match at displayed precision; ML and trading results are independent reconstructions with documented assumptions.</div></header>
<section><h2>What matches the paper</h2>{table(fidelity,0)}<p>All comparisons use independently computed values beside numerical references extracted from the paper. Forecast origins run January 2015–May 2025; realized targets extend through June 2025. Sample sizes are 125, 123, 120, and 114 for the four horizons.</p></section>
<section><h2>Computed forecast results</h2><p>{finding} These rankings reflect {config['forest_trees']}-tree forests and the stated ten-maturity information set. They do not establish that every headline result in the paper replicates.</p><p class="muted">Unweighted average relative RMSE across the forecast horizons and the targets within each segment. Below one beats the random walk.</p>{table(aggregate)}</section>
{panels_html}
<section><h2>Cross-sectional curve fit</h2><p>Mean date-level RMSE over May 1986–June 2025, in percentage points. Pooled decays use the published 0.035/0.006 values. Time-varying optimization is sensitive to unspecified initialization and PSO settings.</p>{table(curve_fit,6)}<div class="notice"><strong>PSO forecasts did not replicate.</strong> The time-varying model's forecasting errors are substantially larger than the paper reports, despite close curve fits. The primary run uses the unscaled objective and stated local tolerances. Unspecified optimizer settings and unstable factor identification remain unresolved.</div><p>The table below compares the primary objective with an explicit objective-scaling sensitivity. Scaling preserves the mathematical optimum but changes effective stopping tolerances. Values are mean relative RMSE across the 13 targets; the other core and fixed-decay models are unchanged.</p>{table(optimizer_sensitivity) if not optimizer_sensitivity.empty else ''}</section>
<section><h2>Trading reconstruction</h2><p>The exercise prices synthetic zero-coupon bonds at entry and their shorter remaining maturities at exit. Slope positions are DV01-neutral at entry and gross-value normalized. Costs are one basis point each way. Signal scaling uses 24 pre-evaluation months.</p><p>Longer-horizon holding periods overlap. Annualized returns below are arithmetic mean holding-period returns multiplied by 12/h; they are not a compounded wealth path. The paper does not fully specify its capital normalization, so these trading numbers are not claimed to reproduce its tables.</p>{table(trading.sort_values(['trade_type','average_rank_total'])) if not trading.empty else '<p>No trading results in this configuration.</p>'}</section>
<section><h2>Why exact ML reproduction remains unresolved</h2><ul><li>The stated ten maturities create 60 yield predictors; Table 4 reports 90 without identifying the other series.</li><li>Appendix A contains 125 codes but the text says 123. Two listed codes do not occur in the source files; those exclusions are logged.</li><li>The macro vintage, missing-value treatment, random seeds, pooled-PSO tuning, and several backtest conventions are unspecified.</li><li>Appendix A's codes 3/6/7 differ from official FRED-MD transformations. Both conventions are implemented. Revised-data publication lags do not remove revision look-ahead.</li></ul><p>The full choices and their justification are in <code>references/REPRODUCTION_DECISIONS.md</code>. Original inputs, configuration, package versions, source hashes, and training dates are retained.</p></section>
<section><h2>Run the 500-tree experiment</h2><pre><code>python -m pip install -r requirements.txt
python run.py --config configs/full_500_trees.json --output results/full_500 --workers 8
python make_report.py --results results/full_500</code></pre><p>The full settings can take several hours. Each completed model/horizon job is checkpointed. Scientific checks cover factor recovery, forecast-label availability, publication lags, training-only imputation, transformation differences, HAC lag selection, bond ageing, and hedge neutrality.</p><p>Use <code>reproduction.ipynb</code> to explore the supplied tables. Each experiment has separate numerical forecasts, model comparisons, feature importances, trading diagnostics, and a provenance manifest.</p></section>
<footer><a href="https://arxiv.org/html/2608.07536v1">Paper</a> · <a href="https://www.discount-bond-data.org/">Original yield data</a> · <a href="https://www.stlouisfed.org/research/economists/mccracken/fred-databases">FRED-MD</a> · Independent implementation, prepared 13 October 2026 UTC.</footer></main></body></html>'''
    (ROOT / "reproduction_report.html").write_text(html)
    notebook(output)
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
