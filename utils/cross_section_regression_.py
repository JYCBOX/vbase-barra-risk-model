from typing import Dict, Union, Tuple, cast
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sim import sim


# === Format Conversion ===

def wide_to_long(
    wide: pd.DataFrame,
    *,
    date_name: str = "date",
    symbol_name: str = "symbol",
    factor_name: str = "factor",
    value_name: str = "loading",
) -> pd.DataFrame:
    """
    Convert a wide factor-exposure table with MultiIndex (factor, symbol)
    into a tidy long DataFrame.
    """
    if not isinstance(wide.columns, pd.MultiIndex) or wide.columns.nlevels != 2:
        raise ValueError("`wide` must have 2-level MultiIndex columns (factor, symbol)")

    long = wide.stack(level=[1, 0]).reset_index()
    long.columns = [date_name, symbol_name, factor_name, value_name]

    return long.sort_values([date_name, symbol_name, factor_name]).reset_index(drop=True)


def long_to_wide(
    long: pd.DataFrame,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
    factor_col: str = "factor",
    value_col: str = "loading",
) -> pd.DataFrame:
    """
    Pivot a tidy long DataFrame into the wide format with MultiIndex(factor, symbol).
    """
    missing = {date_col, symbol_col, factor_col, value_col} - set(long.columns)
    if missing:
        raise ValueError(f"Input `long` missing columns: {missing}")

    wide = long.pivot_table(
        index=date_col,
        columns=[factor_col, symbol_col],
        values=value_col,
        aggfunc="first",
    ).sort_index(axis=1, level=[0, 1])
    wide.index.name = None
    wide.columns.names = [factor_col, symbol_col]
    return wide


# === Single-Period Cross-Sectional Regression (with residuals) ===

def run_cross_sectional_regression(
    asset_returns: pd.Series,
    factor_loadings: pd.DataFrame,
    weights: pd.Series,
    huber_t: float = 1.345,
    return_residual: bool = False,
) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """
    Run a cross-sectional regression for one period using Huber's T norm.
    Optionally return residuals for each asset.
    """
    if asset_returns.empty or factor_loadings.empty or weights.empty:
        raise ValueError("Input series or DataFrames are empty.")

    if not (asset_returns.index.equals(factor_loadings.index) and asset_returns.index.equals(weights.index)):
        raise ValueError("Indices do not match among inputs.")

    com_idx = asset_returns.index.intersection(factor_loadings.index).intersection(weights.index)

    y = cast(pd.Series, asset_returns.loc[com_idx].astype(float))
    x = cast(pd.DataFrame, factor_loadings.loc[com_idx].astype(float))
    w = cast(pd.Series, weights.loc[com_idx].astype(float))

    sqrt_w = np.sqrt(w)
    y_tilde = y * sqrt_w
    X_tilde = x.mul(sqrt_w, axis=0)

    huber = sm.robust.norms.HuberT(t=huber_t)
    model = sm.RLM(endog=y_tilde, exog=X_tilde, M=huber)
    results = model.fit()

    factor_returns = pd.Series(results.params, index=factor_loadings.columns, name="factor_returns")

    if return_residual:
        residuals = pd.Series(results.resid, index=x.index, name="residual")
        return factor_returns, residuals

    return factor_returns


# === Multi-Period Cross-Sectional Regression with Residuals ===

def calculate_factor_returns_with_residuals(
    returns_df: pd.DataFrame,
    exposures_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    huber_t: float = 1.345,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run cross-sectional regression per period and return both factor returns and residuals.

    Parameters
    ----------
    returns_df : DataFrame
        (N_periods, N_assets)
    exposures_df : DataFrame
        (N_periods, MultiIndex[factor, asset])
    weights_df : DataFrame
        (N_periods, N_assets)

    Returns
    -------
    factor_returns_df : DataFrame (N_periods, N_factors)
    residuals_df : DataFrame (N_periods, N_assets)
    """
    for name, df in [("returns", returns_df), ("exposures", exposures_df), ("weights", weights_df)]:
        if df.empty:
            raise ValueError(f"{name}_df is empty")

    returns_df.index = pd.to_datetime(returns_df.index)
    exposures_df.index = pd.to_datetime(exposures_df.index)
    weights_df.index = pd.to_datetime(weights_df.index)

    factor_names = exposures_df.columns.get_level_values(0).unique().tolist()

    raw_data: Dict[str, pd.DataFrame] = {
        "returns": returns_df,
        "exposures": exposures_df,
        "weights": weights_df,
    }

    def callback(
        masked: Dict[str, Union[pd.DataFrame, pd.Series]],
    ) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        ts = masked["returns"].index[-1]

        y = cast(pd.Series, masked["returns"].iloc[-1].dropna())
        exp_ser = cast(pd.Series, masked["exposures"].loc[ts].dropna())
        x = cast(pd.DataFrame, exp_ser.unstack(level=0))
        w = cast(pd.Series, masked["weights"].loc[ts].dropna().astype(float))

        common = [asset for asset in y.index if asset in x.index and asset in w.index]
        if not common:
            return {
                "factor_returns": pd.Series(np.nan, index=factor_names),
                "residuals": pd.Series(np.nan, index=y.index),
            }

        asset_returns = cast(pd.Series, y.loc[common])
        factor_loadings = cast(pd.DataFrame, x.loc[common])
        weights = cast(pd.Series, w.loc[common])

        fr, resid = run_cross_sectional_regression(
            asset_returns=asset_returns,
            factor_loadings=factor_loadings,
            weights=weights,
            huber_t=huber_t,
            return_residual=True
        )

        return {"factor_returns": fr, "residuals": resid}

    out = sim(
        data=cast(Dict[str, Union[pd.DataFrame, pd.Series]], raw_data),
        callback=callback,
        time_index=returns_df.index,
    )

    factor_returns_df = cast(pd.DataFrame, out["factor_returns"])
    residuals_df = cast(pd.DataFrame, out["residuals"])

    return factor_returns_df, residuals_df
