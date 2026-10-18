from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd

MATURITIES = [3, 6, 12, 18, 24, 36, 60, 120, 240, 360]
SPREADS = [(24, 120), (24, 360), (60, 360)]
TARGETS = [str(v) for v in MATURITIES] + [f"{a}_{b}" for a, b in SPREADS]


def load_yields(path):
    """The source already contains annualized continuously compounded DECIMAL yields."""
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index).to_period("M")
    if frame.index.has_duplicates:
        raise ValueError("Duplicate observation months")
    numeric = [c for c in frame if str(c).isdigit()]
    return frame[numeric].astype(float).sort_index()


def make_targets(yields):
    targets = yields[[str(m) for m in MATURITIES]].copy()
    for a, b in SPREADS:
        targets[f"{a}_{b}"] = yields[str(b)] - yields[str(a)]
    return targets


def transform_series(series, code, convention="paper"):
    """Appendix A as written, or the standard FRED-MD convention.

    These conventions disagree for codes 3, 6, and 7. Never treat the two
    conventions as equivalent. No interpolation or future backfill is used.
    """
    code = int(code)
    log = np.log(series.where(series > 0))
    if code == 1:
        return series.copy()
    if code == 2:
        return series.diff()
    if code == 3:
        return series.diff(2) if convention == "paper" else series.diff().diff()
    if code == 4:
        return log
    if code == 5:
        return log.diff()
    if code == 6:
        return log.diff(2) if convention == "paper" else log.diff().diff()
    if code == 7:
        growth = series.pct_change(fill_method=None)
        return growth if convention == "paper" else growth.diff()
    raise ValueError(f"Unknown transformation code {code}")


def load_macro(path, metadata_path, convention="paper"):
    raw = pd.read_csv(path)
    standard_codes = pd.to_numeric(raw.iloc[0, 1:], errors="raise")
    standard_codes.index = raw.columns[1:]
    raw = raw.iloc[1:].set_index(raw.columns[0]).astype(float)
    raw.index = pd.to_datetime(raw.index).to_period("M")
    metadata = pd.read_csv(metadata_path)
    # Keep exact published codes. Unknown identifiers are logged, never silently aliased.
    excluded = metadata.loc[~metadata.code.isin(raw.columns)].copy()
    selected = metadata.loc[metadata.code.isin(raw.columns)].copy()
    columns = {}
    for record in selected.itertuples(index=False):
        code = record.transform if convention == "paper" else standard_codes[record.code]
        columns[record.code] = transform_series(raw[record.code], code, convention)
    transformed = pd.DataFrame(columns, index=raw.index)
    transformed = transformed.replace([np.inf, -np.inf], np.nan)
    return transformed, selected, excluded


def lagged_features(yields, macro=None, factors=None, yield_maturities=None):
    maturities = MATURITIES if yield_maturities is None else yield_maturities
    panels = [yields[[str(m) for m in maturities]].add_prefix("yield_")]
    if factors is not None:
        panels.append(factors.add_prefix("factor_"))
    blocks = []
    for panel in panels:
        blocks += [panel.shift(lag).add_suffix(f"_lag{lag}") for lag in range(1, 7)]
    if macro is not None:
        blocks += [macro.shift(lag).add_prefix("macro_").add_suffix(f"_lag{lag}")
                   for lag in (4, 5, 6)]
    return pd.concat(blocks, axis=1).reindex(yields.index)


def training_positions(index, origin, horizon, window_months, sample_start):
    """Only pairs whose future labels are already observable at the origin."""
    start = max(pd.Period(sample_start, freq="M"), origin - window_months + 1)
    return np.flatnonzero((index >= start) & (index + horizon <= origin))


def impute_training_only(x_train, x_test):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nanmedian(x_train, axis=0)
    medians = np.nan_to_num(medians, nan=0.0)
    return (np.where(np.isnan(x_train), medians, x_train).astype(np.float32),
            np.where(np.isnan(x_test), medians, x_test).astype(np.float32))


def record_provenance(root, config):
    root = Path(root)
    sources = {
        "yield_monthly.csv": "https://drive.google.com/uc?export=download&id=1CdczrQXT6e07ayIczuATtDfYgJN2MLhQ",
        "fred_md_2025_07.csv": "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/historical-vintages-of-fred-md-2015-01-to-2025-12.zip (member FRED-MD_2025m07.csv)",
        "fred_md_2026_07.csv": "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/2026-07-md.csv",
    }
    records = []
    for name, url in sources.items():
        path = root / "data/raw" / name
        if path.exists():
            records.append({"file": str(path.relative_to(root)), "source": url,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "size_bytes": path.stat().st_size})
    return {"paper": "https://arxiv.org/html/2608.07536v1", "retrieved_utc": "2026-10-12",
            "config": config, "sources": records}

