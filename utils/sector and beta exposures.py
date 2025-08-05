# kanav_exposures.py

import pandas as pd
from vbase_utils.stats.pit_robust_betas import pit_robust_betas

def build_sector_exposures(rets_path: str, wiki_url: str) -> pd.DataFrame:
    rets = pd.read_csv(rets_path, index_col=0, parse_dates=True)
    tickers = rets.columns.str.upper()
    table = pd.read_html(wiki_url, header=0)[0][["Symbol", "GICS Sector"]]
    table.columns = ["ticker", "sector"]
    table["ticker"] = table["ticker"].str.upper()
    sector_map = table.set_index("ticker")["sector"]
    aligned = sector_map.reindex(tickers).dropna()
    return pd.get_dummies(aligned).sort_index()

def build_beta_exposures(rets_path: str, half_life: int, min_ts: int) -> pd.DataFrame:
    df = pd.read_csv(rets_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.dropna(axis=1, how="any")
    df_fact = df[["SPY"]]
    df_asset = df.drop(columns=["SPY"])
    month_ends = df.groupby(df.index.to_period("M")).tail(1).index
    res = pit_robust_betas(
        df_asset_rets=df_asset,
        df_fact_rets=df_fact,
        half_life=half_life,
        min_timestamps=min_ts,
        rebalance_time_index=month_ends,
        progress=False,
    )
    return res["df_betas"]

if __name__ == "__main__":
    RETS      = "daily_stock_returns.csv"
    WIKI      = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    HALF_LIFE = 120
    MIN_TS    = 252

    sectors = build_sector_exposures(RETS, WIKI)
    betas   = build_beta_exposures(RETS, HALF_LIFE, MIN_TS)

    sectors.to_csv("sector_factors.csv", index=True)
    betas.to_csv("beta_factors.csv",    index=True)
