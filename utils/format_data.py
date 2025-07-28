import pandas as pd


def normalize_to_date(idx):
    idx = pd.to_datetime(idx, errors="coerce", utc=True).tz_localize(None)
    return idx.normalize()


weights_df = pd.read_csv("../../vbase-data/weights_df.csv", index_col=0)
returns_df = pd.read_csv("../../vbase-data/excess_returns.csv", index_col=0)
expos_df = pd.read_csv("../../vbase-data/factor_loadings_df.csv", header=[0, 1], index_col=0)

cols_to_drop = ["BF.B", "BRK.B"]

mask = ~expos_df.columns.get_level_values(1).isin(cols_to_drop)
expos_df = expos_df.loc[:, mask]

expos_df.to_csv("../../vbase-data/factor_loadings_df.csv")

lvl0, lvl1 = expos_df.columns.levels

sample_lvl0 = list(lvl0)[:5]
if all(isinstance(s, str) and s.isupper() for s in sample_lvl0):
    expos_df.columns = expos_df.columns.swaplevel(0, 1)
    expos_df = expos_df.sort_index(axis=1)
    expos_df.to_csv("../../vbase-data/factor_loadings_df.csv")

tickers_w = set(weights_df.columns)
tickers_r = set(returns_df.columns)
tickers_e = set(expos_df.columns.get_level_values(1))

print(len(tickers_e), len(tickers_w), len(tickers_r))

print(sorted(tickers_e - tickers_w))

valid_tickers = tickers_r & tickers_e
extra_w = sorted(tickers_w - valid_tickers)

if extra_w:
    weights_df = weights_df.drop(columns=extra_w)
    weights_df.to_csv("../../vbase-data/weights_df.csv")


def check_consistency(df1, df2, name1, name2):
    date_diff = df1.index.symmetric_difference(df2.index)

    tick1 = set(df1.columns) if name1 != "factor_loadings" else set(df1.columns.get_level_values(1))
    tick2 = set(df2.columns) if name2 != "factor_loadings" else set(df2.columns.get_level_values(1))
    tkr_diff = tick1.symmetric_difference(tick2)

    ok_dates = len(date_diff) == 0
    ok_tkrs = len(tkr_diff) == 0
    status = "✅" if ok_dates and ok_tkrs else "⛔"
    print(f"{status} {name1} vs {name2} — " f"Date:{ok_dates}  Ticker:{ok_tkrs}")


check_consistency(weights_df, returns_df, "weights_df", "excess_returns")
check_consistency(weights_df, expos_df, "weights_df", "factor_loadings")
check_consistency(returns_df, expos_df, "excess_returns", "factor_loadings")
