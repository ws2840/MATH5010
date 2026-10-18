import zlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .curves import loadings
from .data import MATURITIES, TARGETS, make_targets, lagged_features, training_positions, impute_training_only

MODELS = ["Direct-RF", "Direct-RF-X", "DNS-AR", "DNS-RF", "DNS-RF-X",
          "DNSS-AR", "DNSS-RF", "DNSS-RF-X", "DNSS-AR-PSO", "DNSS-RF-PSO",
          "DNSS-RF-PSO-X", "DNSS-AR-fixedPSO", "DNSS-RF-fixedPSO", "DNSS-RF-fixedPSO-X"]


def stable_seed(seed, *parts):
    return (int(seed) + zlib.crc32("|".join(map(str, parts)).encode())) % (2 ** 32 - 1)


def factor_panel_name(model):
    if "fixedPSO" in model:
        return "DNSS-fixedPSO"
    if "PSO" in model:
        return "DNSS-PSO"
    return "DNS" if model.startswith("DNS-") else "DNSS"


def fit_predict_rf(x_train, target, x_test, trees, max_features, seed):
    model = RandomForestRegressor(n_estimators=trees, min_samples_leaf=5,
                                  max_features=min(int(max_features), x_train.shape[1]),
                                  bootstrap=True, criterion="squared_error", n_jobs=1,
                                  random_state=seed)
    model.fit(x_train, target)
    return float(model.predict(x_test.reshape(1, -1))[0]), model.feature_importances_


def forecast_job(payload):
    model, h, config, yields, macro, panels, max_features = payload
    targets = make_targets(yields)
    if model.startswith("Direct"):
        factors, panel = None, None
        response = targets
    else:
        panel = panels[factor_panel_name(model)]
        factors = panel["factors"]
        response = factors
    is_ar = "-AR" in model
    # Standard direct-horizon AR uses beta_t to forecast beta_{t+h}. This
    # interpretation matches every published baseline AR relative-RMSE cell.
    features = (factors.shift(config.get("ar_origin_lag", 0)) if is_ar else
                lagged_features(yields, macro if model.endswith("-X") else None,
                                factors, config["yield_predictor_maturities"]))
    x_values = features.to_numpy()
    future_response = response.shift(-h)
    changes = future_response - response
    index = yields.index
    first_origin = pd.Period(config["oos_origin_start"], freq="M")
    last_origin = pd.Period(config["oos_origin_end"], freq="M")
    calibration_start = first_origin - config["signal_calibration_months"]
    forecasts, importances, audit = [], [], []
    name_for_m = model.replace("-fixedPSO", "")
    m_lookup = max_features[max_features.model == name_for_m].set_index("target")
    candidate_origins = index[(index >= calibration_start) & (index <= last_origin)]
    if config.get("origin_stride", 1) > 1:
        candidate_origins = candidate_origins[::config["origin_stride"]]
    for origin in candidate_origins:
        if origin + h > index[-1]:
            continue
        pos = index.get_loc(origin)
        train_pos = training_positions(index, origin, h, config["window_months"], config["sample_start"])
        # All yield/factor labels must be present. Macro holes are imputed within this window only.
        train_pos = train_pos[np.isfinite(future_response.iloc[train_pos].to_numpy()).all(axis=1)]
        if len(train_pos) < 36:
            raise ValueError(f"Insufficient training observations at {origin}")
        assert (index[train_pos] + h <= origin).all()
        split = "calibration" if origin < first_origin else "oos"
        audit.append({"model": model, "horizon": h, "origin": str(origin), "split": split,
                      "train_pairs": len(train_pos), "train_base_first": str(index[train_pos[0]]),
                      "train_base_last": str(index[train_pos[-1]]),
                      "train_label_last": str(index[train_pos[-1]] + h),
                      "predictor_count": features.shape[1]})
        if not is_ar:
            x_train, x_test = impute_training_only(x_values[train_pos], x_values[pos])
        predicted_response = []
        for i, target_name in enumerate(response.columns):
            if is_ar:
                predictor = x_values[train_pos, i]
                good = np.isfinite(predictor)
                design = np.column_stack([np.ones(good.sum()), predictor[good]])
                coef = np.linalg.lstsq(design, future_response.iloc[train_pos, i].to_numpy()[good], rcond=None)[0]
                prediction = float(coef @ np.array([1., x_values[pos, i]]))
            else:
                m = int(m_lookup.loc[str(target_name), f"h{h}"])
                delta, importance = fit_predict_rf(x_train, changes.iloc[train_pos, i].to_numpy(), x_test,
                                                  config["forest_trees"], m,
                                                  stable_seed(config["seed"], model, h, target_name, origin))
                prediction = response.iloc[pos, i] + delta
                if split == "oos":
                    importances.append({"model": model, "horizon": h, "origin": str(origin),
                                        "response": target_name, "importance": importance})
            predicted_response.append(prediction)
        if model.startswith("Direct"):
            predicted_targets = dict(zip(TARGETS, predicted_response))
        else:
            l1, l2 = panel["decays"][pos]
            predicted_yields = loadings(MATURITIES, l1, None if np.isnan(l2) else l2) @ predicted_response
            predicted_targets = dict(zip(map(str, MATURITIES), predicted_yields))
            for a, b in ((24, 120), (24, 360), (60, 360)):
                predicted_targets[f"{a}_{b}"] = predicted_targets[str(b)] - predicted_targets[str(a)]
        for target_name in TARGETS:
            forecasts.append({"model": model, "horizon": h, "origin": str(origin),
                              "target_month": str(origin + h), "target": target_name, "split": split,
                              "current": targets.loc[origin, target_name],
                              "predicted": predicted_targets[target_name],
                              "realized": targets.loc[origin + h, target_name]})
    # Collapse importances only after retaining equal weights across origins for each response/horizon.
    importance_records = []
    for response_name in response.columns:
        selected = [r["importance"] for r in importances if r["response"] == response_name]
        if selected:
            mean = np.mean(selected, axis=0)
            importance_records += [{"model": model, "horizon": h, "response": str(response_name),
                                    "feature": feature, "importance": float(value)}
                                   for feature, value in zip(features.columns, mean)]
    return pd.DataFrame(forecasts), pd.DataFrame(importance_records), pd.DataFrame(audit)


def random_walk_records(yields, config):
    targets = make_targets(yields)
    rows = []
    for h in config["horizons"]:
        for origin in targets.index:
            if str(origin) < config["oos_origin_start"] or str(origin) > config["oos_origin_end"]:
                continue
            if origin + h not in targets.index:
                continue
            for target in TARGETS:
                rows.append({"model": "Random Walk", "horizon": h, "origin": str(origin),
                             "target_month": str(origin + h), "target": target, "split": "oos",
                             "current": targets.loc[origin, target], "predicted": targets.loc[origin, target],
                             "realized": targets.loc[origin + h, target]})
    return pd.DataFrame(rows)

