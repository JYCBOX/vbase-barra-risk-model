import pandas as pd

weights_df = pd.read_csv("../../vbase-data/weights/weights20220131.csv", index_col=0, header=None)
factors_df = pd.read_csv(
    "../../vbase-data/factor_loadings/factor_loadings20220131.csv", index_col=0
)

print(len(set(weights_df.index)))
print(len(set(factors_df.index)))
print(list(set(factors_df.index) - set(weights_df.index)))

common_tickers = list(set(weights_df.index) & set(factors_df.index))

rets_df = pd.read_csv("../../vbase-data/us_stocks_1d_rets.csv", index_col=0, parse_dates=True)

valid_tickers = [tkr for tkr in common_tickers if tkr in rets_df.columns]
excess_rets = rets_df[valid_tickers].subtract(rets_df["SPY"], axis=0)
excess_rets = excess_rets[sorted(excess_rets.columns)]

excess_rets.to_csv("../../vbase-data/excess_returns.csv")
