"""
Robust cross-sectional regression with Huber-T to estimate factor returns & residuals
from daily returns, factor exposures (MultiIndex columns), and weights.

Run:
    python robust_xsec_regression.py

Outputs:
    factor_returns_{YYYYMMDD}.csv
    residuals_{YYYYMMDD}.csv
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from datetime import datetime

# ============
# FILE PATHS (edit if needed)
# ============
RETURNS_CSV   = "us_stocks_1d_rets.csv"     # daily returns (T x N_assets)
EXPOSURES_CSV = "factor_loadings_df.csv"    # factor exposures, wide MultiIndex (factor, asset)
WEIGHTS_CSV   = "weights_df.csv"            # regression weights (T x N_assets)
OUT_DIR       = "."                         # output directory

# ============
# CONFIGURATION
# ============
HUBER_T = 1.345                 # Huber-T parameter (robustness tuning)
ADD_INTERCEPT = True            # add intercept column to regression
SECTOR_FACTORS = [              # list of sector dummy factor names
    "Sector_Comm",
    "Sector_ConsDisc",
    "Sector_ConsStap",
    "Sector_Energy",
    "Sector_Financials",
    "Sector_HealthCare",
    "Sector_Industrials",
    "Sector_IT",
    "Sector_Materials",
    "Sector_RealEstate",
    "Sector_Utilities",
]
SECTOR_SUM_TO_ZERO = True       # enforce sum-to-zero constraint across sector dummies
STANDARDIZE_EXPOSURES = False   # standardize factor exposures (z-score) before regression
MIN_OBS_BUFFER = 2              # minimum observations = #params + buffer
USE_COMMON_UNIVERSE = True      # restrict to assets present in all three dataframes
DROP_ALL_NAN_COLS_AFTER = True  # drop assets that are all-NaN in residuals_df

# ============
# FORMAT CONVERSION HELPERS
# ============

def wide_to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Convert wide MultiIndex exposures into tidy long format."""
    if not isinstance(wide.columns, pd.MultiIndex) or wide.columns.nlevels != 2:
        raise ValueError("`wide` must have 2-level MultiIndex columns (factor, symbol)")
    long = wide.stack(level=[1, 0]).reset_index()
    long.columns = ["date", "symbol", "factor", "loading"]
    return long.sort_values(["date", "symbol", "factor"]).reset_index(drop=True)


def long_to_wide(long: pd.DataFrame) -> pd.DataFrame:
    """Convert tidy long exposures into wide MultiIndex format."""
    required = {"date", "symbol", "factor", "loading"}
    if not required.issubset(long.columns):
        raise ValueError(f"Input must contain {required}, got {set(long.columns)}")
    wide = long.pivot_table(
        index="date",
        columns=["factor", "symbol"],
        values="loading",
        aggfunc="first",
    ).sort_index(axis=1, level=[0, 1])
    wide.index.name = None
    wide.columns.names = ["factor", "symbol"]
    return wide

# ============
# UTILITIES
# ============

def _normalize_dates_index(idx: pd.Index) -> pd.DatetimeIndex:
    """Convert tz-aware or mixed date index into naive daily dates."""
    di = pd.to_datetime(idx, utc=True, errors="coerce")
    if di.isna().any():
        fallback = pd.to_datetime(idx, errors="coerce")
        di = di.where(~di.isna(), fallback)
        try:
            if getattr(di, "tz", None) is not None:
                di = di.tz_convert(None)
        except Exception:
            try:
                di = di.tz_localize(None)
            except Exception:
                pass
    else:
        di = di.tz_convert(None)
    return pd.to_datetime(di.date)

# ============
# CORE REGRESSION
# ============

def run_cross_sectional_regression(
    asset_returns: pd.Series,
    factor_loadings: pd.DataFrame,
    weights: pd.Series,
    *,
    huber_t: float = HUBER_T,
    add_intercept: bool = ADD_INTERCEPT,
    sector_factors: Optional[List[str]] = None,
    sector_sum_to_zero: bool = SECTOR_SUM_TO_ZERO,
    standardize_exposures: bool = STANDARDIZE_EXPOSURES,
    min_obs: Optional[int] = None,
    return_residual: bool = False,
) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """Run one-period robust regression using Huber-T norm."""
    # Align indexes
    common = asset_returns.index.intersection(factor_loadings.index).intersection(weights.index)
    if len(common) == 0:
        idx = (["intercept"] if add_intercept else []) + list(factor_loadings.columns)
        fr = pd.Series(np.nan, index=idx)
        return (fr, pd.Series(dtype=float)) if return_residual else fr

    # Extract aligned data
    y = asset_returns.loc[common].astype(float)
    X = factor_loadings.loc[common].astype(float)
    w = weights.loc[common].astype(float).clip(lower=1e-12)

    # Drop missing rows
    mask = y.notna() & w.notna()
    mask &= ~X.isna().any(axis=1)
    y, X, w = y[mask], X[mask], w[mask]
    if y.empty:
        idx = (["intercept"] if add_intercept else []) + list(factor_loadings.columns)
        fr = pd.Series(np.nan, index=idx)
        resid = pd.Series(np.nan, index=asset_returns.index)
        return (fr, resid) if return_residual else fr

    # Apply sum-to-zero on sector dummies
    if sector_sum_to_zero and sector_factors:
        exist = [c for c in sector_factors if c in X.columns]
        if exist:
            X.loc[:, exist] = X.loc[:, exist] - X.loc[:, exist].mean(axis=0)

    # Optional standardization
    if standardize_exposures:
        X = (X - X.mean(axis=0)) / X.std(axis=0).replace(0, np.nan)

    # Add intercept if required
    col_order = list(X.columns)
    if add_intercept:
        X = sm.add_constant(X, has_constant="add")
        col_order = ["const"] + col_order

    # Check min obs
    if min_obs is None:
        min_obs = X.shape[1] + MIN_OBS_BUFFER
    if len(y) < min_obs:
        idx = (["intercept"] if add_intercept else []) + list(factor_loadings.columns)
        fr = pd.Series(np.nan, index=idx)
        resid = pd.Series(np.nan, index=asset_returns.index)
        return (fr, resid) if return_residual else fr

    # Weighted robust regression
    sw = np.sqrt(w.values)
    y_tilde = y.values * sw
    X_tilde = X.values * sw[:, None]
    huber = sm.robust.norms.HuberT(t=huber_t)
    results = sm.RLM(endog=y_tilde, exog=X_tilde, M=huber).fit()

    # Factor returns
    fr = pd.Series(results.params, index=col_order, name="factor_returns")
    if add_intercept:
        fr.index = ["intercept"] + list(factor_loadings.columns)
    else:
        fr.index = list(factor_loadings.columns)

    # Residuals
    if return_residual:
        y_hat = pd.Series((X.values @ results.params), index=y.index)
        resid = y - y_hat
        resid_full = pd.Series(np.nan, index=asset_returns.index, name="residual")
        resid_full.loc[resid.index] = resid
        return fr, resid_full

    return fr


def calculate_factor_returns_with_residuals(
    returns_df: pd.DataFrame,
    exposures_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    *,
    huber_t: float = HUBER_T,
    add_intercept: bool = ADD_INTERCEPT,
    sector_factors: Optional[List[str]] = None,
    sector_sum_to_zero: bool = SECTOR_SUM_TO_ZERO,
    standardize_exposures: bool = STANDARDIZE_EXPOSURES,
    min_obs_per_period: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run regression for each date and return factor returns + residuals."""
    # Normalize dates
    returns_df.index   = _normalize_dates_index(returns_df.index)
    exposures_df.index = _normalize_dates_index(exposures_df.index)
    weights_df.index   = _normalize_dates_index(weights_df.index)

    # Restrict to common asset universe if configured
    if USE_COMMON_UNIVERSE:
        aset_ret = set(returns_df.columns)
        aset_w   = set(weights_df.columns)
        aset_exp = set(exposures_df.columns.get_level_values(1))
        universe = sorted(aset_ret & aset_w & aset_exp)
        returns_df   = returns_df[universe]
        weights_df   = weights_df[universe]
        exposures_df = exposures_df.loc[:, exposures_df.columns.get_level_values(1).isin(universe)]

    dates = returns_df.index.intersection(exposures_df.index).intersection(weights_df.index).sort_values()
    if dates.empty:
        raise ValueError("No overlapping dates among returns/exposures/weights.")

    factor_names = exposures_df.columns.get_level_values(0).unique().tolist()
    all_assets = returns_df.columns.tolist()

    fr_rows, resid_rows = [], []
    for ts in dates:
        y = returns_df.loc[ts].dropna()
        exp_ser = exposures_df.loc[ts].dropna()
        X = exp_ser.unstack(level=0).reindex(columns=factor_names)
        w = weights_df.loc[ts].dropna().astype(float)

        common = y.index.intersection(X.index).intersection(w.index)
        if len(common) == 0:
            fr_rows.append(pd.Series(np.nan, index=(["intercept"] if add_intercept else []) + factor_names, name=ts))
            resid_rows.append(pd.Series(np.nan, index=all_assets, name=ts))
            continue

        fr, resid = run_cross_sectional_regression(
            asset_returns=y.loc[common],
            factor_loadings=X.loc[common],
            weights=w.loc[common],
            huber_t=huber_t,
            add_intercept=add_intercept,
            sector_factors=sector_factors,
            sector_sum_to_zero=sector_sum_to_zero,
            standardize_exposures=standardize_exposures,
            min_obs=min_obs_per_period,
            return_residual=True,
        )

        order = (["intercept"] if add_intercept else []) + factor_names
        fr = fr.reindex(order)

        resid_full = pd.Series(np.nan, index=all_assets, name=ts)
        resid_full.loc[resid.index] = resid

        fr.name = ts
        fr_rows.append(fr)
        resid_rows.append(resid_full)

    factor_returns_df = pd.DataFrame(fr_rows).sort_index()
    residuals_df = pd.DataFrame(resid_rows).sort_index()

    if DROP_ALL_NAN_COLS_AFTER:
        residuals_df = residuals_df.loc[:, residuals_df.notna().any(axis=0)]

    return factor_returns_df, residuals_df

# ============
# MAIN SCRIPT
# ============

def main():
    print("[1/5] Loading data...")
    returns_df  = pd.read_csv(RETURNS_CSV, index_col=0)
    weights_df  = pd.read_csv(WEIGHTS_CSV, index_col=0)
    exposures_df = pd.read_csv(EXPOSURES_CSV, header=[0, 1], index_col=0)  # assumes wide MultiIndex

    print("[2/5] Normalizing indices...")
    returns_df.index   = _normalize_dates_index(returns_df.index)
    weights_df.index   = _normalize_dates_index(weights_df.index)
    exposures_df.index = _normalize_dates_index(exposures_df.index)

    print("[3/5] Running regressions...")
    fr_df, resid_df = calculate_factor_returns_with_residuals(
        returns_df=returns_df,
        exposures_df=exposures_df,
        weights_df=weights_df,
        huber_t=HUBER_T,
        add_intercept=ADD_INTERCEPT,
        sector_factors=SECTOR_FACTORS,
        sector_sum_to_zero=SECTOR_SUM_TO_ZERO,
        standardize_exposures=STANDARDIZE_EXPOSURES,
    )

    # Use YYYYMMDD only (no HHMMSS)
    ts = datetime.now().strftime("%Y%m%d")
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    fr_path    = out_dir / f"factor_returns_{ts}.csv"
    resid_path = out_dir / f"residuals_{ts}.csv"

    print("[4/5] Saving outputs...")
    fr_df.to_csv(fr_path, float_format="%.10g")
    resid_df.to_csv(resid_path, float_format="%.10g")

    print("[5/5] Done.")
    print(f"  Factor returns -> {fr_path}")
    print(f"  Residuals      -> {resid_path}")

if __name__ == "__main__":
    main()
