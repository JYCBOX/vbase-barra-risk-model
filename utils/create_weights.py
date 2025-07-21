from pathlib import Path

import numpy as np
import pandas as pd

df = pd.read_csv("../../vbase-data/market_caps.csv", parse_dates=[0])
df.drop(df.columns[1], axis=1, inplace=True)
df.rename(columns={df.columns[0]: "Date"}, inplace=True)
df.set_index("Date", inplace=True)

df = df.loc["2022-01-31":]

print(df.head(5))

sqrt_cap = np.sqrt(df)
weights = sqrt_cap.div(sqrt_cap.sum(axis=1), axis=0)

out_dir = Path("../../vbase-data/weights")
out_dir.mkdir(exist_ok=True)

for day, row in weights.iterrows():
    # day is a Timestamp, row is a Series indexed by ticker
    filename = out_dir / f"weights{day.strftime('%Y%m%d')}.csv"
    row.to_frame(name="weight").to_csv(filename)

print(f"Done! {len(weights)} files saved to {out_dir.resolve()}")
