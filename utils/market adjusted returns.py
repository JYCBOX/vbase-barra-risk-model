import pandas as pd

date = "20190103"

w = pd.read_csv(f"weights{date}.csv", index_col=0, header=None, squeeze=True)
f = pd.read_csv(f"factor_loadings{date}.csv", index_col=0)
common = f.index.intersection(w.index)

r = pd.read_csv("daily_stock_returns.csv", index_col=0, parse_dates=True)
symbols = [s for s in common if s in r.columns]

mkt_adj = r[symbols].subtract(r["SPY"], axis=0)
mkt_adj.to_csv("market_adjusted_returns.csv")
