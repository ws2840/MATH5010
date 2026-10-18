#!/usr/bin/env python3
"""Check temporal, numerical, and accounting invariants of a completed run."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from yieldcurve.data import MATURITIES, TARGETS, load_yields, make_targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/replication_50trees")
    parser.add_argument("--run-unit-tests", action="store_true")
    args = parser.parse_args()
    output = ROOT / args.results
    manifest = json.loads((output / "manifest.json").read_text())
    config = manifest["config"]
    checks = []

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append({"check": name, "result": "PASS"})

    for source in manifest["sources"]:
        check(f"Input checksum: {source['file']}", hashlib.sha256((ROOT / source["file"]).read_bytes()).hexdigest() == source["sha256"])
    for name, expected in manifest["source_sha256"].items():
        check(f"Source checksum: {name}", hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected)
    forecasts = pd.read_csv(output / "forecasts.csv", dtype={"target": str})
    metrics = pd.read_csv(output / "forecast_metrics.csv", dtype={"target": str})
    audit = pd.read_csv(output / "training_audit.csv")
    check("All predictions and observations are finite", np.isfinite(forecasts[["current", "predicted", "realized"]].to_numpy()).all())
    check("No duplicate forecast keys", not forecasts.duplicated(["model", "horizon", "origin", "target"]).any())
    origins = pd.PeriodIndex(forecasts.origin, freq="M")
    destinations = pd.PeriodIndex(forecasts.target_month, freq="M")
    check("Every forecast destination has the stated horizon", (destinations.asi8 - origins.asi8 == forecasts.horizon.to_numpy()).all())
    check("No training labels after their forecast origins", (pd.PeriodIndex(audit.train_label_last, freq="M") <= pd.PeriodIndex(audit.origin, freq="M")).all())
    check("All configured model/horizon/target metrics exist", len(metrics) == (len(config["models"]) + 1) * len(config["horizons"]) * len(TARGETS))
    yields = load_yields(ROOT / "data/raw/yield_monthly.csv").loc[:"2025-06"]
    targets = make_targets(yields)
    for h in config["horizons"]:
        end = min(pd.Period(config["oos_origin_end"], freq="M"), yields.index[-1] - h)
        n = len(pd.period_range(config["oos_origin_start"], end, freq="M"))
        check(f"Complete aligned OOS sample for horizon {h}", (metrics.loc[metrics.horizon == h, "n"] == n).all())
    check("Current values agree with original yield source", np.allclose(
        forecasts.current.to_numpy(), [targets.loc[o, t] for o, t in zip(origins, forecasts.target)], atol=1e-14, rtol=0))
    check("Realizations agree with original yield source", np.allclose(
        forecasts.realized.to_numpy(), [targets.loc[o, t] for o, t in zip(destinations, forecasts.target)], atol=1e-14, rtol=0))
    rw = forecasts[forecasts.model == "Random Walk"]
    check("Random walk uses only the origin observation", np.allclose(rw.predicted, rw.current, atol=1e-14, rtol=0))
    check("RMSEs and DM p-values are finite", np.isfinite(metrics[["rmse_pp", "relative_rmse", "p_value"]].to_numpy()).all())
    check("DM p-values lie in [0,1]", metrics.p_value.between(0, 1).all())
    factor = forecasts[~forecasts.model.str.startswith("Direct")]
    wide = factor.pivot(index=["model", "horizon", "origin"], columns="target", values="predicted")
    for a, b in [(24, 120), (24, 360), (60, 360)]:
        check(f"Factor/RW {a}-{b} slopes equal predicted yield differences",
              np.allclose(wide[f"{a}_{b}"], wide[str(b)] - wide[str(a)], atol=1e-14, rtol=0))
    trade_path = output / "trades.csv"
    if trade_path.exists():
        trades = pd.read_csv(trade_path)
        check("Trading returns and scaling estimates are finite", np.isfinite(trades[["position", "sigma_in_sample", "total_net", "roll_down", "curve_move_net"]].to_numpy()).all())
        check("Trading exposures respect the cap", (trades.position.abs() <= 1 + 1e-12).all())
        check("Net trading accounting reconciles", np.allclose(trades.total_net, trades.roll_down + trades.curve_move_net, atol=1e-12, rtol=0))
        check("Each round trip uses the stated transaction cost", np.allclose(trades.transaction_cost, trades.position.abs() * config["transaction_cost_one_way_bp"] * 2e-4, atol=1e-14, rtol=0))
    unit_tests = {"result": "not_run_by_validator", "command": "python -m unittest discover -s tests -v"}
    if args.run_unit_tests:
        completed = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                                   cwd=ROOT, capture_output=True, text=True)
        test_output = completed.stdout + completed.stderr
        (output / "unit_test_output.txt").write_text(test_output)
        if completed.returncode:
            raise AssertionError("Scientific unit tests failed; see unit_test_output.txt")
        match = re.search(r"Ran (\d+) tests?", test_output)
        unit_tests.update({"result": "PASS", "count": int(match.group(1)) if match else None})
    summary = {"result": "PASS", "checks": checks, "forecast_rows": len(forecasts),
               "metric_rows": len(metrics), "scientific_unit_tests": unit_tests}
    (output / "validation_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"PASS: {len(checks)} result invariants; {len(forecasts):,} forecasts; {len(metrics)} metrics.")


if __name__ == "__main__":
    main()
