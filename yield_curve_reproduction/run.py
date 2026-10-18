#!/usr/bin/env python3
import os
# One computational thread per worker avoids nested parallelism.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from yieldcurve.data import MATURITIES, load_yields, make_targets, load_macro, record_provenance
from yieldcurve.curves import build_curve_panels, loadings
from yieldcurve.forecast import MODELS, forecast_job, random_walk_records
from yieldcurve.evaluate import forecast_metrics, compare_to_paper, paired_comparisons
from yieldcurve.trading import backtest


def initial_data_checks(yields, panels, output):
    targets = make_targets(yields).loc["1986-05":"2025-06"] * 100
    summary = targets.agg(["mean", "std", "min", "max"]).T
    summary["25th pct."] = targets.quantile(.25)
    summary["Median"] = targets.median()
    summary = summary.rename(columns={"mean": "Mean", "std": "Std. Dev.", "min": "Min", "max": "Max"})
    summary.index.name = "Maturity / Measure"
    paper = pd.read_csv(ROOT / "references/paper_summary_statistics.csv", dtype={"Maturity / Measure": str}).set_index("Maturity / Measure")
    comparisons = []
    for target in summary.index:
        for stat in paper.columns:
            actual, expected = float(summary.loc[target, stat]), float(paper.loc[target, stat])
            comparisons.append({"target": target, "statistic": stat, "reproduced": actual, "paper_value": expected,
                                "difference": actual - expected, "rounds_to_paper": round(actual, 3) == expected})
    pd.DataFrame(comparisons).to_csv(output / "summary_statistics_comparison.csv", index=False)
    summary.to_csv(output / "summary_statistics.csv")
    records = []
    for name, panel in panels.items():
        errors = []
        mask = yields.index >= pd.Period("1986-05", freq="M")
        for i in np.flatnonzero(mask):
            l1, l2 = panel["decays"][i]
            fitted = loadings(MATURITIES, l1, None if np.isnan(l2) else l2) @ panel["factors"].iloc[i].to_numpy()
            errors.append(float(100 * np.sqrt(np.mean((fitted - yields.iloc[i][list(map(str, MATURITIES))].to_numpy()) ** 2))))
        expected = {"DNS": .1140, "DNSS": .0923, "DNSS-PSO": .0284, "DNSS-PSO-ZLB": .0331}.get(name)
        records.append({"model": name, "mean_cross_sectional_rmse_pp": float(np.mean(errors)),
                        "paper_value": expected, "difference": np.mean(errors) - expected if expected is not None else np.nan})
    pd.DataFrame(records).to_csv(output / "curve_fit_comparison.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Reproduce arXiv:2608.07536v1 with explicit assumptions")
    parser.add_argument("--config", default="configs/completed_run.json")
    parser.add_argument("--output", default="results/completed_run")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--trees", type=int)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--horizons", nargs="+", type=int)
    parser.add_argument("--skip-pso", action="store_true")
    parser.add_argument("--skip-trading", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    if args.trees is not None:
        config["forest_trees"] = args.trees
    if args.models:
        config["models"] = args.models
    if args.horizons:
        config["horizons"] = args.horizons
    if args.skip_pso:
        config["models"] = [m for m in config["models"] if "-PSO" not in m]
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    provenance = record_provenance(ROOT, config)
    provenance["package_versions"] = {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib")}
    source_files = list(ROOT.glob("*.py")) + list((ROOT / "src").rglob("*.py")) + list((ROOT / "tests").glob("*.py"))
    provenance["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(source_files)}
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Configuration, source code, packages, or input files changed. Choose a new --output directory.")
    provenance["fingerprint"] = fingerprint
    manifest.write_text(json.dumps(provenance, indent=2))
    yields = load_yields(ROOT / "data/raw/yield_monthly.csv").loc[:"2025-06"]
    yields = yields.loc[yields[list(map(str, MATURITIES))].notna().all(axis=1)]
    if not yields.index.equals(pd.period_range(yields.index[0], yields.index[-1], freq="M")):
        raise ValueError("Missing calendar month in yield sample")
    macro, selected, excluded = load_macro(ROOT / "data/raw" / config["macro_file"],
                                           ROOT / "references/macro_variables.csv", config["macro_transformation"])
    selected.to_csv(output / "selected_macro_variables.csv", index=False)
    excluded.to_csv(output / "excluded_macro_variables.csv", index=False)
    macro.reindex(yields.index).to_csv(output / "transformed_macro.csv", index_label="month")
    max_features = pd.read_csv(ROOT / "references/rf_max_features.csv", dtype={"target": str})
    panels = build_curve_panels(yields, MATURITIES, config["pso"], output,
                               include_pso=any("-PSO" in m for m in config["models"]))
    initial_data_checks(yields, panels, output)
    frames = [random_walk_records(yields, config)]
    importance_frames, audit_frames = [], []
    payloads = []
    for model in config["models"]:
        if model not in MODELS:
            raise ValueError(f"Unknown model {model}")
        for h in config["horizons"]:
            payloads.append((model, h, config, yields, macro, panels, max_features))
    start = time.monotonic()
    pending = []
    for payload in payloads:
        key = f"{payload[0]}_h{payload[1]}"
        forecast_path = checkpoints / f"{key}_forecasts.csv"
        if forecast_path.exists():
            frames.append(pd.read_csv(forecast_path, dtype={"target": str}))
            importance_path = checkpoints / f"{key}_importance.csv"
            if importance_path.exists():
                importance_frames.append(pd.read_csv(importance_path, dtype={"response": str}))
            audit_frames.append(pd.read_csv(checkpoints / f"{key}_audit.csv"))
            print(f"Resumed {key}", flush=True)
        else:
            pending.append(payload)
    print(f"Forecasting {len(pending)} model/horizon jobs; {config['forest_trees']} trees; {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(forecast_job, payload): payload[:2] for payload in pending}
        completed = 0
        for future in as_completed(futures):
            model, h = futures[future]
            frame, importance, audit = future.result()
            key = f"{model}_h{h}"
            # Forecast is written last so a checkpoint denotes a complete job.
            audit.to_csv(checkpoints / f"{key}_audit.csv", index=False)
            if not importance.empty:
                importance.to_csv(checkpoints / f"{key}_importance.csv", index=False)
                importance_frames.append(importance)
            frame.to_csv(checkpoints / f"{key}_forecasts.csv", index=False)
            frames.append(frame)
            audit_frames.append(audit)
            completed += 1
            print(f"Completed {completed}/{len(pending)}: {key}; elapsed {time.monotonic() - start:.0f}s", flush=True)
    forecasts = pd.concat(frames, ignore_index=True)
    forecasts = forecasts.sort_values(["model", "horizon", "origin", "target"])
    forecasts.to_csv(output / "forecasts.csv", index=False)
    importance = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    if not importance.empty:
        importance.to_csv(output / "feature_importance.csv", index=False)
    pd.concat(audit_frames, ignore_index=True).to_csv(output / "training_audit.csv", index=False)
    metrics = forecast_metrics(forecasts)
    metrics.to_csv(output / "forecast_metrics.csv", index=False)
    reference = pd.read_csv(ROOT / "references/paper_forecast_metrics.csv", dtype={"target": str})
    compare_to_paper(metrics, reference).to_csv(output / "paper_comparison.csv", index=False)
    paired_comparisons(forecasts).to_csv(output / "paired_model_comparisons.csv", index=False)
    if not args.skip_trading:
        trades, trade_metrics, summary = backtest(forecasts, yields, config)
        trades.to_csv(output / "trades.csv", index=False)
        trade_metrics.to_csv(output / "trading_metrics.csv", index=False)
        summary.to_csv(output / "trading_summary.csv", index=False)
    print(f"Finished. Results: {output}; elapsed forecast/evaluation {time.monotonic() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
