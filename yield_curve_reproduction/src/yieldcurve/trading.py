"""Synthetic bond exercise; conventions missing from the paper are explicit here."""
import numpy as np
import pandas as pd

from .data import SPREADS


def bond_price(yield_decimal, maturity_months):
    return np.exp(-np.asarray(yield_decimal) * maturity_months / 12)


def bond_dv01(yield_decimal, maturity_months):
    return bond_price(yield_decimal, maturity_months) * (maturity_months / 12) * 1e-4


def bond_returns(yields, origin, maturity, horizon):
    if maturity <= horizon:
        raise ValueError("The aged bond must have positive remaining maturity")
    initial = bond_price(yields.loc[origin, str(maturity)], maturity)
    rolled = bond_price(yields.loc[origin, str(maturity - horizon)], maturity - horizon)
    terminal = bond_price(yields.loc[origin + horizon, str(maturity - horizon)], maturity - horizon)
    total, roll = terminal / initial - 1, rolled / initial - 1
    return float(total), float(roll), float(total - roll)


def steepener_weights(yields, origin, a, b):
    pa = bond_price(yields.loc[origin, str(a)], a)
    pb = bond_price(yields.loc[origin, str(b)], b)
    qa = bond_dv01(yields.loc[origin, str(b)], b) / bond_dv01(yields.loc[origin, str(a)], a)
    # qa is a FACE-AMOUNT hedge ratio. Convert cash P&L to a gross-market-value return.
    gross_value = qa * pa + pb
    return float(qa), 1., float(gross_value)


def slope_returns(yields, origin, a, b, horizon):
    ra, rda, rca = bond_returns(yields, origin, a, horizon)
    rb, rdb, rcb = bond_returns(yields, origin, b, horizon)
    qa, qb, gross = steepener_weights(yields, origin, a, b)
    pa = bond_price(yields.loc[origin, str(a)], a)
    pb = bond_price(yields.loc[origin, str(b)], b)
    return tuple(float((qa * pa * left - qb * pb * right) / gross)
                 for left, right in ((ra, rb), (rda, rdb), (rca, rcb)))


def momentum_signal(yields, origin, maturity):
    excess = []
    for j in range(1, 13):
        past = origin - j
        total, _, _ = bond_returns(yields, past, maturity, 1)
        cash_return = 1 / bond_price(yields.loc[past, "1"], 1) - 1
        excess.append(total - cash_return)
    return float(np.sum(excess))


def backtest(forecasts, yields, config):
    first = pd.Period(config["oos_origin_start"], freq="M")
    cal_months = pd.period_range(first - config["signal_calibration_months"], first - 1, freq="M")
    fee_round_trip = 2 * config["transaction_cost_one_way_bp"] * 1e-4
    targets = ["24", "120", "360", "24_120", "24_360", "60_360"]
    records = []

    def append_strategy(model, h, target, signals, sigma, origins):
        for origin, signal in zip(origins, signals):
            scale = 1. if model == "Random Walk" else float(np.clip(signal / max(sigma, 1e-12), -1, 1))
            if "_" in target:
                a, b = map(int, target.split("_"))
                total, roll, curve = slope_returns(yields, origin, a, b, h)
                trade_type = "slope"
            else:
                total, roll, curve = bond_returns(yields, origin, int(target), h)
                trade_type = "duration"
            cost = abs(scale) * fee_round_trip
            records.append({"model": model, "horizon": h, "target": target,
                            "trade_type": trade_type, "origin": str(origin), "position": scale,
                            "sigma_in_sample": sigma, "total_gross": scale * total,
                            "roll_down": scale * roll, "curve_move": scale * curve,
                            "transaction_cost": cost, "total_net": scale * total - cost,
                            "curve_move_net": scale * curve - cost})

    for (model, h, target), group in forecasts[forecasts.target.isin(targets)].groupby(["model", "horizon", "target"]):
        if model == "Random Walk":
            if "_" in target:
                continue
            test = group[group.split == "oos"].sort_values("origin")
            append_strategy(model, h, target, np.ones(len(test)), 1., pd.PeriodIndex(test.origin, freq="M"))
            continue
        sign = 1 if "_" in target else -1
        calibrate = group[group.split == "calibration"]
        sigma = float((calibrate.predicted - calibrate.current).std(ddof=1))
        test = group[group.split == "oos"].sort_values("origin")
        origins = pd.PeriodIndex(test.origin, freq="M")
        append_strategy(model, h, target, sign * (test.predicted - test.current).to_numpy(), sigma, origins)
    # Benchmarks share identical holding-period dates with model forecasts.
    for h in config["horizons"]:
        origins = yields.index[(yields.index >= first) &
                               (yields.index <= pd.Period(config["oos_origin_end"], freq="M")) &
                               (yields.index + h <= yields.index[-1])]
        for target in targets:
            slope = "_" in target
            def roll_signal(month):
                if slope:
                    a, b = map(int, target.split("_"))
                    return slope_returns(yields, month, a, b, h)[1]
                return bond_returns(yields, month, int(target), h)[1]
            cal = np.array([roll_signal(m) for m in cal_months])
            append_strategy("Roll-down", h, target, [roll_signal(m) for m in origins], float(cal.std(ddof=1)), origins)
            if not slope:
                cal = np.array([momentum_signal(yields, m, int(target)) for m in cal_months])
                signals = [momentum_signal(yields, m, int(target)) for m in origins]
                append_strategy("TSMOM", h, target, signals, float(cal.std(ddof=1)), origins)
    trades = pd.DataFrame(records)
    metrics = []
    for (model, h, target, kind), group in trades.groupby(["model", "horizon", "target", "trade_type"]):
        ret = group.total_net
        # Arithmetic annualization of overlapping holding-period observations.
        # This is a diagnostic, not a non-overlapping portfolio equity curve.
        metrics.append({"model": model, "horizon": h, "target": target, "trade_type": kind,
                        "n": len(group), "annual_return_pct": float(100 * 12 / h * ret.mean()),
                        "annual_roll_down_pct": float(100 * 12 / h * group.roll_down.mean()),
                        "annual_curve_move_net_pct": float(100 * 12 / h * group.curve_move_net.mean()),
                        "holding_period_sharpe": float(ret.mean() / ret.std(ddof=1)) if ret.std(ddof=1) else np.nan,
                        "hit_rate": float((ret > 0).mean()), "average_absolute_position": float(group.position.abs().mean())})
    metrics = pd.DataFrame(metrics)
    metrics["rank_total"] = metrics.groupby(["horizon", "target"])["annual_return_pct"].rank(ascending=False, method="min")
    metrics["rank_curve"] = metrics.groupby(["horizon", "target"])["annual_curve_move_net_pct"].rank(ascending=False, method="min")
    summary = metrics.groupby(["model", "trade_type"]).agg(
        average_rank_total=("rank_total", "mean"), average_rank_curve=("rank_curve", "mean"),
        annual_return_pct=("annual_return_pct", "mean"), annual_curve_move_net_pct=("annual_curve_move_net_pct", "mean"),
        annual_roll_down_pct=("annual_roll_down_pct", "mean"), hit_rate=("hit_rate", "mean"))
    return trades, metrics, summary.reset_index()

