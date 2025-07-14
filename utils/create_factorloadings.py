from pathlib import Path

import pandas as pd

beta_path = Path("../../vbase-data/beta_factors.csv")
sector_path = Path("../../vbase-data/sector_factors.csv")
out_dir = Path("../../vbase-data")

beta = pd.read_csv(beta_path, index_col=0, parse_dates=True)
sector = pd.read_csv(sector_path, index_col=0).astype(int)

for date, beta_row in beta.iterrows():
    exposures = beta_row.dropna()
    common_tickers = exposures.index.intersection(sector.index)
    if common_tickers.empty:
        continue

    df = pd.DataFrame(index=common_tickers)
    df["market"] = 1
    df["beta"] = exposures.loc[common_tickers]
    df = df.join(sector, how="left")

    out_file = out_dir / f"factor_loadings{date.strftime('%Y%m%d')}.csv"
    df.to_csv(out_file)
