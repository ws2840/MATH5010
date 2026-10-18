# Reproduction decisions

The implementation is independent. The paper does not provide enough information
to identify every experiment uniquely. The following choices are exposed rather
than treated as exact replications of unknown author code.

| Item | Evidence and implementation |
|---|---|
| Yield data | Original Filipović–Pelger–Ye monthly observations and monthly maturity grid. Data through June 2025. Decimal, annualized, continuously compounded yields. The 78 displayed Table 1 statistics match at the paper's precision. |
| Evaluation dates | Treat 2015-01 to 2025-05 as **forecast origins**, and retain only observed future targets through June 2025. This matches all 52 random-walk absolute RMSE cells to two decimal places. There are 125, 123, 120, and 114 observations at horizons 1, 3, 6, and 12. |
| Rolling estimation | 344 calendar months, the length of May 1986–December 2014. A training pair is eligible only if its target month is at or before the forecast origin. Historical features may precede the left edge of the training window. |
| AR indexing | Fit a separate direct-horizon regression `beta[t+h] = intercept + coefficient * beta[t]` for each factor. This recovers all 104 DNS-AR and DNSS-AR relative-RMSE cells at two decimals. Reading the paper's “first lag” literally as `beta[t-1]` does not: it introduces an extra month. `ar_origin_lag=1` permits that sensitivity. |
| RF yield information | The stated ten maturities, lags 1–6: **60** yield predictors. Table 4 instead reports 90, implying 15 series, which the text does not identify. No five additional predictors are invented. The maturity list can be changed explicitly in config. |
| Factor RF information | Include the ten yield histories plus all factor histories, lags 1–6, following §4.3. Table 6's abbreviated descriptions omit the yield histories. The resulting DNS and DNSS counts are 78 and 84, versus 108 and 114 in Table 4. |
| Macro series | Appendix A lists **125** codes, whereas §3.2 says 123. Exactly 123 listed identifiers exist in the supplied FRED-MD files. Exclude and log `CES2000000007` and `SP500`; no silent alias to `S&P 500` is made. This matches the stated 369 lagged macro predictors, but does not establish the author's exact selection. |
| Macro vintage | The paper does not state its FRED-MD vintage. Primary run uses archived July 2025; July 2026 is also supplied for sensitivity. Both are revised-data panels, not a sequence of real-time vintages. A publication lag does not eliminate revision look-ahead. |
| Macro transformations | Primary run implements Appendix A **as written**. For codes 3, 6, and 7 this differs from the official FRED-MD convention. The official sensitivity uses the transformation codes in the raw FRED-MD header. It also changes BUSLOANS from the appendix's code 2 to the raw header's code 6. |
| Missing macro observations | Median computed within each eligible training window; entirely missing training columns use zero. The paper gives no imputation rule. No future interpolation or backfill. |
| RF hyperparameters | Reuse target-specific `max_features` from Table 4; 5 observations per leaf; bootstrap samples. The completed run uses **50 trees**; the full config uses the paper's **500 trees**. Seeds and library implementation are unreported in the paper, so a fixed independent seed is used. |
| RF tuning | Published Table 4 choices are reused, not re-optimized. The optional tuner implements recursive validation over target dates 2010–2014 with the stated candidate grid. Pooled-PSO RF settings are not separately tabulated; use baseline DNSS RF settings and disclose this. |
| PSO search | 300 particles, at most 1,000 iterations, 10 local refinements, stated bounds and tolerances. Adaptive ring neighborhoods, previous-date warm start, OLS anchor. Inertia 0.9→0.4, acceleration 2/2, velocity cap 20% of range, and a 50-iteration stall count are **assumptions** because those values are unreported. |
| Optimizer objective scale | Primary and 500-tree configurations use **unscaled decimal-yield SSE**, so the stated local-search tolerances act on that objective. An explicit `objective_scale=1e6` sensitivity is also supplied: scaling preserves the mathematical minimizer but changes numerical convergence and can materially destabilize factor forecasts. |
| Ordered local search | Reparameterize the second decay as a fraction of the first to maintain ordering under L-BFGS-B. For the optional ZLB calibration use nonnegative long/short endpoints, with the short endpoint capped at 1. That numerical bound is an additional assumption and is nonbinding for ordinary Treasury rates. |
| PSO factor extraction | Re-extract coefficients by unrestricted conditional OLS, as stated in §4.3.1. This can remove the optimizer's coefficient/ZLB bounds. Those fits are separate from the bounded optimizer's coefficients. |
| Fixed/pooled PSO | Use the published decays 0.035 and 0.006 from Figure 10. The original pooling interval and calibration procedure are not specified. Their prospective availability cannot be established. |
| HAC/DM | Largest significant ACF lag up to floor(T^(1/3)), Bartlett confidence bands. Bartlett/Newey–West HAC weighting and two-sided asymptotic-normal p-values; the paper does not state kernel, sidedness, or a finite-sample correction. |
| Signal volatility | Standard deviation of 24 strictly pre-evaluation monthly forecast signals. Early calibration windows contain all available pairs from May 1986 rather than unavailable 30-year yield history. Sample standard deviation with ddof=1. |
| Duration trading | Synthetic zero-coupon bond ages by h months. Signals clipped to [-1,1]. Random walk is a constant long position. Roll-down and 12-month excess-return momentum are scaled on the pre-evaluation period. |
| Slope trading | Face amounts satisfy entry DV01 neutrality. Normalize P&L by gross entry market value and cap the scaled signal at one in magnitude. The paper does not fully specify the capital denominator/capping for slope positions; these are economically explicit independent choices. |
| Trading costs | 1 bp each way on entry gross exposure; round-trip cost is 2 bp times absolute signal. Exit-notional versus entry-notional cost accounting is unspecified in the paper. |
| Annualization | Arithmetic annualized mean holding-period return, `mean(R_h) * 12/h`. Report unannualized holding-period Sharpe. Monthly entries overlap at h>1; no independent-observation assumption or compounded portfolio equity curve is claimed. |

## Sources

- [Paper, version 1](https://arxiv.org/html/2608.07536v1)
- [Original Treasury data and conventions](https://www.discount-bond-data.org/)
- [FRED-MD data and vintage archive](https://www.stlouisfed.org/research/economists/mccracken/fred-databases)
- [Official transformation definitions, Appendix](https://files.stlouisfed.org/files/htdocs/wp/2015/2015-012.pdf)

Reported tables, RF choices, and macro metadata are stored as factual reference
CSV files. Raw inputs, configuration, software versions, and SHA-256 fingerprints
are recorded in the results manifest. Recomputed results are stored separately
from the paper's reported numbers.
