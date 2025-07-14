import pandas as pd

path = "../../vbase-data/us_stocks_1d_rets.csv"
URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

df = pd.read_csv(path, index_col=0, parse_dates=True)

sector_map = pd.read_html(URL, header=0)[0][["Symbol", "GICS Sector"]]
sector_map.columns = ["ticker", "sector"]
sector_map["ticker"] = sector_map["ticker"].str.upper()
sector_map = sector_map.set_index("ticker")["sector"]

tickers = df.columns.str.upper()
sectors = sector_map.reindex(tickers)
sector_factors = pd.get_dummies(sectors[sectors.notna()])
sector_factors.to_csv("sector_factors.csv")

print(f"Allocate {sector_factors.shape[0]} stocks in {sector_factors.shape[1]} sectors")
