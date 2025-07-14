import pandas as pd
from vbase_utils.stats.pit_robust_betas import pit_robust_betas

path = "../../vbase-data/us_stocks_1d_rets.csv"

df = pd.read_csv(path, index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index, utc=True, errors="raise").tz_convert("America/New_York")

df = df.dropna(axis=1, how="any")

df_fact_rets = df[["SPY"]].copy()
df_asset_rets = df.drop(columns=["SPY"])

# Parameters for beta calculation, half-life and minimum timestamps
HALF_LIFE = 120
MIN_TS = 252

# Create a monthly rebalance time index
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
