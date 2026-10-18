import numpy as np
import pandas as pd
from scipy.stats import norm


def dm_test(benchmark_errors, model_errors, lag=None):
    d = np.asarray(benchmark_errors) ** 2 - np.asarray(model_errors) ** 2
    n = len(d)
    centered = d - d.mean()
    gamma0 = centered @ centered / n
    if gamma0 < 1e-25:
        return {"dm": 0., "p_value": 1., "hac_lag": 0}
    # np.cbrt avoids floor(125**(1/3)) incorrectly becoming 4 through rounding.
    cap = min(int(np.floor(np.cbrt(n) + 1e-12)), n - 1)
    correlations = [1.] + [float(centered[k:] @ centered[:-k] / (n * gamma0)) for k in range(1, cap + 1)]
    if lag is None:
        significant = []
        for k in range(1, cap + 1):
            band = 1.96 * np.sqrt((1 + 2 * np.sum(np.square(correlations[1:k]))) / n)
            if abs(correlations[k]) > band:
                significant.append(k)
        lag = max(significant, default=0)
    lrv = gamma0
    for k in range(1, lag + 1):
        gamma = centered[k:] @ centered[:-k] / n
        lrv += 2 * (1 - k / (lag + 1)) * gamma
    stat = float(d.mean() / np.sqrt(max(lrv, 1e-25) / n))
    return {"dm": stat, "p_value": float(2 * norm.sf(abs(stat))), "hac_lag": lag}


def forecast_metrics(forecasts):
    oos = forecasts[forecasts.split == "oos"].copy()
    oos["error"] = oos.realized - oos.predicted
    records = []
    for (model, h, target), group in oos.groupby(["model", "horizon", "target"]):
        rw = oos[(oos.model == "Random Walk") & (oos.horizon == h) & (oos.target == target)]
        aligned = group.merge(rw[["origin", "error"]], on="origin", suffixes=("", "_rw"))
        rmse = np.sqrt(np.mean(aligned.error ** 2))
        rw_rmse = np.sqrt(np.mean(aligned.error_rw ** 2))
        stats = dm_test(aligned.error_rw, aligned.error)
        stars = 3 if stats["p_value"] < .01 else 2 if stats["p_value"] < .05 else 1 if stats["p_value"] < .1 else 0
        records.append({"model": model, "horizon": h, "target": target, "n": len(aligned),
                        "rmse_pp": float(100 * rmse), "relative_rmse": float(rmse / rw_rmse),
                        "stars": stars, **stats})
    return pd.DataFrame(records)


def compare_to_paper(metrics, reference):
    long = metrics.melt(id_vars=["model", "horizon", "target", "n"],
                        value_vars=["rmse_pp", "relative_rmse"], var_name="metric", value_name="reproduced")
    joined = long.merge(reference, on=["model", "horizon", "target", "metric"])
    joined["difference"] = joined.reproduced - joined.value
    joined["rounds_to_paper"] = joined.reproduced.round(2) == joined.value
    return joined.rename(columns={"value": "paper_value"})


def paired_comparisons(forecasts):
    oos = forecasts[forecasts.split == "oos"].copy()
    oos["error"] = oos.realized - oos.predicted
    models = set(oos.model)
    pairs = [(m, m[:-2]) for m in models if m.endswith("-X") and m[:-2] in models]
    for model in sorted(models):
        if model.startswith("DNS"):
            benchmark = "Direct-RF-X" if model.endswith("-X") else "Direct-RF"
            if benchmark in models:
                pairs.append((model, benchmark))
        if "PSO" in model:
            benchmark = model.replace("-fixedPSO", "").replace("-PSO", "")
            if benchmark in models:
                pairs.append((model, benchmark))
    records = []
    for model, benchmark in pairs:
        for (h, target), group in oos[oos.model == model].groupby(["horizon", "target"]):
            base = oos[(oos.model == benchmark) & (oos.horizon == h) & (oos.target == target)]
            aligned = group.merge(base[["origin", "error"]], on="origin", suffixes=("", "_benchmark"))
            ratio = np.sqrt(np.mean(aligned.error ** 2) / np.mean(aligned.error_benchmark ** 2))
            records.append({"model": model, "benchmark": benchmark, "horizon": h,
                            "target": target, "relative_rmse": float(ratio),
                            **dm_test(aligned.error_benchmark, aligned.error)})
    return pd.DataFrame(records)

