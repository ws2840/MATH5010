#!/usr/bin/env python3
"""Optional recursive validation, independent of the paper's supplied m choices."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from yieldcurve.data import MATURITIES, load_yields, load_macro, make_targets, lagged_features, training_positions, impute_training_only
from yieldcurve.curves import extract_factors
from yieldcurve.forecast import stable_seed, fit_predict_rf


def candidate_grid(p):
    return sorted(set(max(1, min(p, int(round(v)))) for v in
                      [np.sqrt(p), p / 10, p / 6, p / 4, p / 3, p / 2, 2 * p / 3]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["Direct-RF", "Direct-RF-X", "DNS-RF", "DNS-RF-X", "DNSS-RF", "DNSS-RF-X"], default="Direct-RF-X")
    parser.add_argument("--horizon", type=int, choices=[1, 3, 6, 12], default=1)
    parser.add_argument("--targets", nargs="+", default=["3"])
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--output", default="results/tuning.csv")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/full_500_trees.json").read_text())
    yields = load_yields(ROOT / "data/raw/yield_monthly.csv").loc[:"2025-06"]
    yields = yields.loc[yields[list(map(str, MATURITIES))].notna().all(axis=1)]
    macro, _, _ = load_macro(ROOT / "data/raw" / config["macro_file"], ROOT / "references/macro_variables.csv", config["macro_transformation"])
    factors = None
    if args.model.startswith("Direct"):
        responses = make_targets(yields)
    else:
        factors = extract_factors(yields, MATURITIES, .0609, None if args.model.startswith("DNS-") else .01)
        responses = factors
    features = lagged_features(yields, macro if args.model.endswith("-X") else None, factors)
    x = features.to_numpy()
    changes = responses.shift(-args.horizon) - responses
    records = []
    for target in args.targets:
        if target not in responses:
            raise ValueError(f"Unknown response {target}; choices: {list(responses)}")
        for m in candidate_grid(features.shape[1]):
            errors = []
            for target_month in pd.period_range("2010-01", "2014-12", freq="M"):
                origin = target_month - args.horizon
                pos = yields.index.get_loc(origin)
                train = training_positions(yields.index, origin, args.horizon, config["window_months"], config["sample_start"])
                a, b = impute_training_only(x[train], x[pos])
                change, _ = fit_predict_rf(a, changes.iloc[train][target].to_numpy(), b, args.trees, m,
                                          stable_seed(config["seed"], args.model, args.horizon, target, origin))
                errors.append(responses.loc[target_month, target] - responses.loc[origin, target] - change)
            rmse = float(np.sqrt(np.mean(np.square(errors))))
            records.append({"model": args.model, "horizon": args.horizon, "target": target,
                            "predictor_count": features.shape[1], "trees": args.trees,
                            "max_features": m, "validation_rmse_decimal": rmse, "n_validation": 60})
            print(f"{args.model} h={args.horizon} {target}: m={m}, validation RMSE={rmse:.6f}", flush=True)
    result = pd.DataFrame(records)
    result["chosen"] = result.groupby("target").validation_rmse_decimal.transform("min") == result.validation_rmse_decimal
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)


if __name__ == "__main__":
    main()
