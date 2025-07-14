import pandas as pd
import numpy as np
from typing import Optional, Dict
from statsmodels.regression.linear_model import WLS
from statsmodels.tools.tools import add_constant


def robust_betas(
    df_asset_rets: pd.DataFrame,
    df_fact_rets: pd.DataFrame,
    half_life: Optional[float] = None,
    lambda_: Optional[float] = None,
    min_timestamps: int = 10,
) -> pd.DataFrame:
    betas = pd.DataFrame(index=df_fact_rets.columns, columns=df_asset_rets.columns, dtype=float)

    for asset in df_asset_rets.columns:
        y = df_asset_rets[asset].dropna()
        x = df_fact_rets.loc[y.index].dropna()
        common_idx = y.index.intersection(x.index)
        if len(common_idx) < min_timestamps:
            continue
        y = y.loc[common_idx]
        X = x.loc[common_idx]

        if half_life:
            decay = 0.5 ** (1 / half_life)
            weights = pd.Series([decay**i for i in reversed(range(len(X)))], index=common_idx)
        elif lambda_:
            weights = pd.Series([lambda_**i for i in reversed(range(len(X)))], index=common_idx)
        else:
            weights = pd.Series(1.0, index=common_idx)

        X = add_constant(X)
        model = WLS(y, X, weights=weights)
        results = model.fit()
        betas[asset] = results.params.drop("const")

    return betas


def pit_robust_betas(
    df_asset_rets: pd.DataFrame,
    df_fact_rets: pd.DataFrame,
    half_life: Optional[float] = None,
    lambda_: Optional[float] = None,
    min_timestamps: int = 10,
    rebalance_time_index: Optional[pd.DatetimeIndex] = None,
    progress: bool = True,
) -> Dict[str, pd.DataFrame]:
    if rebalance_time_index is None:
        rebalance_time_index = df_asset_rets.index

    asset_names = df_asset_rets.columns
    factor_names = df_fact_rets.columns
    betas_list = []

    for i, date in enumerate(rebalance_time_index):
        if progress:
            print(f"Processing {i+1}/{len(rebalance_time_index)}: {date}")

        df_ret_hist = df_asset_rets.loc[:date].dropna(how="all")
        df_fact_hist = df_fact_rets.loc[:date].dropna(how="all")
        valid_idx = df_ret_hist.index.intersection(df_fact_hist.index)

        df_ret_hist = df_ret_hist.loc[valid_idx]
        df_fact_hist = df_fact_hist.loc[valid_idx]

        if len(valid_idx) < min_timestamps:
            continue

        betas = robust_betas(
            df_ret_hist,
            df_fact_hist,
            half_life=half_life,
            lambda_=lambda_,
            min_timestamps=min_timestamps,
        )
        betas.index.name = "factor"
        betas.columns.name = "asset"
        betas["timestamp"] = date
        betas_list.append(betas.reset_index().set_index(["timestamp", "factor"]))

    df_betas = pd.concat(betas_list).sort_index()

    # === 🔧 Add this part: create full panel + ffill ===
    full_index = pd.MultiIndex.from_product(
        [rebalance_time_index, factor_names],
        names=["timestamp", "factor"]
    )
    df_betas = df_betas.reindex(full_index)
    df_betas = df_betas.sort_index().ffill()

    return {"df_betas": df_betas}



# === Main script ===
path = "us_stocks_1d_rets.csv"

df = pd.read_csv(path, index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index, utc=True, errors="raise").tz_convert("America/New_York")

df = df.dropna(axis=1, how="any")
df_fact_rets = df[["SPY"]].copy()
df_asset_rets = df.drop(columns=["SPY"])

HALF_LIFE = 120
MIN_TS = 252

month_end_idx = df.groupby(df.index.to_period("M")).tail(1).index

res_monthly = pit_robust_betas(
    df_asset_rets=df_asset_rets,
    df_fact_rets=df_fact_rets,
    half_life=HALF_LIFE,
    min_timestamps=MIN_TS,
    rebalance_time_index=month_end_idx,
    progress=True,
)

df_betas_monthly = res_monthly["df_betas"]
df_betas_monthly.to_csv("beta_factors.csv")
print("✓ Finished calculating and saving beta_factors.csv")
