# Yield-curve paper reproduction

Independent Python implementation of [arXiv:2608.07536v1](https://arxiv.org/html/2608.07536v1),
*Yield Curve Prediction with Machine Learning: Forecasting Approaches and the Role
of Macroeconomic Predictors* (Jeron Tan Kang, 2026).

**The supplied results use 50 trees, versus 500 in the paper.** They are a completed
reduced-tree replication, not an exact reproduction of all reported ML or trading
tables. Exact author code, predictor definitions, macro vintage, and several
backtest conventions are unavailable. All assumptions are listed in
`references/REPRODUCTION_DECISIONS.md`.

## Start here

Open `reproduction_report.html` for the completed results, or
`reproduction.ipynb` in VS Code/Jupyter to explore the data and tables. No data
download is required: the original public input snapshots are included.
To execute the notebook, use a Python kernel with `ipykernel` installed.

Create a Python 3.12 environment and install the recorded dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

On Windows, activate the environment with `.venv\Scripts\activate`.

Run the full 500-tree configuration:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run.py \
  --config configs/full_500_trees.json --output results/full_500 --workers 8
python make_report.py --results results/full_500
python validate_results.py --results results/full_500 --run-unit-tests
```

On Windows, omit the leading environment-variable assignments. A full run can
take several hours depending on hardware. The program checkpoints completed
model/horizon jobs and resumes them when rerun with the same configuration,
source code, package versions, and input bytes. Use a new output directory when
any of those change.

Recompute the supplied 50-tree experiment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run.py \
  --config configs/completed_run.json --output results/my_50_tree_run --workers 8
python make_report.py --results results/my_50_tree_run
python validate_results.py --results results/my_50_tree_run --run-unit-tests
```

Run only the baseline models for a faster study:

```bash
python run.py --models Direct-RF Direct-RF-X DNS-AR DNS-RF DNS-RF-X \
  DNSS-AR DNSS-RF DNSS-RF-X --skip-pso --output results/core --workers 8
```

The default output path is `results/completed_run`; specify a fresh path to avoid
mixing new computations with the supplied results in `results/replication_50trees`.

## What is implemented

- Random walk and all 14 competing direct, DNS, DNSS, time-varying PSO, and fixed/pooled-PSO model variants.
- Separate single-output forests for each response and horizon; 5 observations per leaf; published target-specific `max_features` settings.
- Forecast origins January 2015–May 2025, horizons 1/3/6/12 months, a 344-month rolling window, and explicit exclusion of unavailable future training labels.
- Ten maturity targets and three separately forecast slope targets for direct models. Factor-model slopes are reconstructed from predicted maturity yields.
- 123 macro series and lags 4–6; observed yield and factor lags 1–6. Training-window-only missing-value imputation.
- Hybrid PSO–L-BFGS-B fitting, conditional OLS factor extraction, and a separate ZLB cross-sectional fit.
- Absolute/relative RMSE, HAC Diebold–Mariano tests, paired model comparisons, and impurity-based feature importance.
- A synthetic-bond duration/slope trading exercise with aged bonds, pre-evaluation signal scaling, gross-normalized DV01-neutral slope positions, and round-trip costs.

## Numerical fidelity

The original yield data reproduce all **78 Table 1 summary-statistic cells** at
published precision. The inferred origin-based evaluation schedule reproduces
all **52 random-walk RMSE cells**, and standard direct-horizon AR indexing
reproduces all **104 baseline factor-AR relative-RMSE cells** and all **104
baseline factor-AR absolute-RMSE cells** at two decimals.
The report and comparison CSVs give additional checks and actual ML differences.
**The time-varying PSO forecasts did not replicate:** their errors are substantially
larger than reported, even after using the unscaled objective and stated local
tolerances. Better curve fit does not establish a faithful forecast replication.

The paper's predictor counts do not match its stated maturity list. This code
uses the explicitly stated ten maturities: 60 direct yield predictors, or 78/84
when DNS/DNSS factor histories are included. With 369 macro predictors these
become 429/447/453. Table 4 reports larger dimensions; this prevents a uniquely
identified forest replication.

The main macro panel is the archived July 2025 vintage. This is a revised-data
experiment with a publication lag, not a real-time-vintage experiment.
`configs/official_fred_sensitivity.json` implements official FRED transformations
because Appendix A differs for codes 3/6/7. To study the July 2026 snapshot,
change `macro_file` in a copy of the config and use a fresh output directory.
The official sensitivity also uses the source header's per-series code assignments,
including code 6 for BUSLOANS versus code 2 in the appendix.

## Files and units

| File | Contents |
|---|---|
| `data/raw/` | Original Treasury and FRED-MD source snapshots |
| `references/` | Paper's numerical tables, macro metadata, RF choices, and decisions |
| `src/yieldcurve/` | Data, curve extraction, forecasts, evaluation, and trading modules |
| `results/replication_50trees/manifest.json` | Configuration, input/source hashes, and software versions |
| `forecast_metrics.csv` | RMSE in percentage points, relative RMSE, DM p-values, counts |
| `paper_comparison.csv` | Recomputed values beside paper values; signed gaps and rounded matches |
| `forecasts.csv` | Origin, target month, realized/predicted/current values in decimal units |
| `training_audit.csv` | Actual training dates, last observable label, predictor count |
| `curve_fit_comparison.csv` | Mean date-level curve-fit RMSE in percentage points |
| `optimizer_scaled_sensitivity_forecast_metrics.csv` | Earlier optimizer-scale sensitivity, with all other settings held fixed |
| `paired_model_comparisons.csv` | Macro, direct-versus-factor, and PSO comparisons |
| `feature_importance.csv` | Origin-averaged impurity importance for each response/horizon |
| `trades.csv`, `trading_metrics.csv`, `trading_summary.csv` | Independent backtest diagnostics |
| `figures/` | Scientific plots produced from computed values |

Paths in the last eight rows are relative to the results directory. A yield of
0.04 means 4%; an RMSE of 0.20 percentage points means 20 basis points. Do not
substitute constant-maturity **par** yields for these zero-coupon yields.

## Optional recursive tuning

The primary experiment reuses Table 4 settings. To retune one response with the
stated recursive 2010–2014 validation design and the actual predictor dimension:

```bash
python tune.py --model Direct-RF-X --horizon 1 --targets 3 --trees 500
```

`tune.py` currently covers the fixed-decay core forest models. The default example
tunes one response because exhaustive recursive tuning is expensive. This tuning
run was not used to generate the supplied results.

## Backtest interpretation

The economic exercise is an independent reconstruction. Capital normalization,
annualization, transaction-cost accounting, and signal calibration conventions
are made explicit. Holding-period observations overlap at longer horizons.
Reported Sharpe values are holding-period mean/standard deviation, and the
annualized arithmetic mean is not a compounded portfolio wealth path. The
computed trading tables must not be described as the author's exact results.

## Optimizer sensitivity

The primary run uses unscaled decimal-yield SSE and the stated local-search
tolerances. `configs/scaled_optimizer_sensitivity.json` multiplies the local
objective by one million. That preserves the mathematical optimum but tightens
effective numerical stopping criteria. The earlier sensitivity results are
retained to show how unstable a time-varying factor representation can become.
This sensitivity does not change the core direct/DNS/DNSS or fixed-decay models.
