import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yieldcurve.data import transform_series, training_positions, lagged_features, impute_training_only, MATURITIES
from yieldcurve.curves import loadings, extract_factors, fit_nss_pso
from yieldcurve.forecast import fit_predict_rf
from yieldcurve.evaluate import dm_test
from yieldcurve.trading import bond_returns, bond_price, bond_dv01, steepener_weights, slope_returns


class ScientificChecks(unittest.TestCase):
    def test_ols_recovers_exact_synthetic_curve(self):
        betas = np.array([.045, -.02, .035, -.01])
        y = loadings(MATURITIES, .0609, .01) @ betas
        frame = pd.DataFrame([y], columns=list(map(str, MATURITIES)))
        np.testing.assert_allclose(extract_factors(frame, MATURITIES, .0609, .01).iloc[0], betas, atol=1e-12)

    def test_unscaled_optimizer_improves_a_known_fit_and_obeys_bounds(self):
        tau = np.asarray(MATURITIES, dtype=float)
        a1 = (1 - np.exp(-.028 * tau)) / (.028 * tau)
        c1 = a1 - np.exp(-.028 * tau)
        c2 = (1 - np.exp(-.004 * tau)) / (.004 * tau) - np.exp(-.004 * tau)
        observed = .035 - .020 * a1 + .014 * c1 + .030 * c2
        settings = {"particles": 300, "iterations": 1000, "local_particles": 10,
                    "stall_tolerance": 1e-8, "stall_iterations": 50,
                    "inertia_start": .9, "inertia_end": .4, "c1": 2., "c2": 2.,
                    "velocity_fraction": .2, "objective_scale": 1.}
        fitted, bounded, _ = fit_nss_pso(observed, MATURITIES, settings, seed=7)
        reconstructed = loadings(MATURITIES, fitted[-2], fitted[-1]) @ fitted[:4]
        reference_basis = loadings(MATURITIES, .0609, .01)
        reference_fit = reference_basis @ np.linalg.lstsq(reference_basis, observed, rcond=None)[0]
        self.assertLess(np.sum((observed - reconstructed) ** 2), np.sum((observed - reference_fit) ** 2))
        self.assertTrue((np.abs(bounded[:4]) <= 1 + 1e-12).all())
        self.assertGreaterEqual(bounded[-2], bounded[-1])
        self.assertGreater(bounded[-1], 0)

    def test_macro_code_six_distinguishes_both_conventions(self):
        series = pd.Series(np.exp([0., 1., 3., 6.]))
        np.testing.assert_allclose(transform_series(series, 6, "paper").iloc[2:], [3., 5.])
        np.testing.assert_allclose(transform_series(series, 6, "official").iloc[2:], [1., 1.])

    def test_publication_lags_exclude_recent_macro_data(self):
        months = pd.period_range("2000-01", periods=18, freq="M")
        yields = pd.DataFrame(np.ones((18, 10)), index=months, columns=list(map(str, MATURITIES)))
        macro = pd.DataFrame({"x": np.arange(18.)}, index=months)
        origin = months[12]
        before = lagged_features(yields, macro).loc[origin]
        macro.loc[origin - 3:] = 999999.
        after = lagged_features(yields, macro).loc[origin]
        np.testing.assert_allclose(before, after)

    def test_training_labels_and_predictions_cannot_use_future(self):
        months = pd.period_range("2000-01", periods=80, freq="M")
        rng = np.random.default_rng(7)
        y = rng.normal(size=80)
        x = rng.normal(size=(80, 4))
        origin, h = months[60], 12
        ids = training_positions(months, origin, h, 36, "2000-01")
        self.assertTrue((months[ids] + h <= origin).all())
        response = y[ids + h] - y[ids]
        prediction, _ = fit_predict_rf(x[ids], response, x[60], 20, 2, 42)
        y[61:] = 1e9
        response_after = y[ids + h] - y[ids]
        prediction_after, _ = fit_predict_rf(x[ids], response_after, x[60], 20, 2, 42)
        self.assertEqual(prediction, prediction_after)

    def test_imputation_uses_only_training_values(self):
        train = np.array([[1., np.nan], [3., 2.]])
        test = np.array([np.nan, 1000.])
        completed, query = impute_training_only(train, test)
        self.assertEqual(query[0], 2.)
        self.assertEqual(completed[0, 1], 2.)

    def test_flat_unchanged_curve_has_only_carry_roll_return(self):
        months = pd.period_range("2000-01", periods=14, freq="M")
        yields = pd.DataFrame(.05, index=months, columns=[str(m) for m in range(1, 361)])
        total, roll, curve = bond_returns(yields, months[0], 120, 12)
        self.assertAlmostEqual(total, np.exp(.05) - 1, places=12)
        self.assertAlmostEqual(roll, total, places=12)
        self.assertAlmostEqual(curve, 0., places=12)

    def test_steepener_has_zero_entry_dv01_and_correct_sign(self):
        months = pd.period_range("2000-01", periods=2, freq="M")
        yields = pd.DataFrame(.04, index=months, columns=[str(m) for m in range(1, 361)])
        qa, qb, gross = steepener_weights(yields, months[0], 24, 120)
        self.assertAlmostEqual(qa * bond_dv01(.04, 24), qb * bond_dv01(.04, 120), places=12)
        yields.loc[months[1], "119"] += .01
        total, roll, curve = slope_returns(yields, months[0], 24, 120, 1)
        self.assertGreater(curve, 0.)
        self.assertAlmostEqual(total, roll + curve, places=12)

    def test_dm_identical_forecasts_has_no_significance(self):
        e = np.linspace(-1, 1, 100)
        self.assertEqual(dm_test(e, e)["p_value"], 1.)

    def test_hac_allows_fifth_lag_with_125_observations(self):
        # A known oscillating loss differential has strong lag-5 dependence.
        d = np.sin(2 * np.pi * np.arange(125) / 10)
        benchmark = np.sqrt(5 + d)
        competing = np.full(125, np.sqrt(5.))
        self.assertEqual(dm_test(benchmark, competing)["hac_lag"], 5)


if __name__ == "__main__":
    unittest.main()
