import os

import numpy as np
import pandas as pd

DATA_DIR = "../vbase-data"
WEIGHTS_DIR = os.path.join(DATA_DIR, "weights")
TOL = 1e-8


def weighted_variance(x: pd.Series, w: pd.Series) -> float:

    w = w.reindex(x.index).fillna(0.0)
    total_w = w.sum()
    if total_w == 0:
        return np.nan
    w = w / total_w
    mu = (x * w).sum()
    return ((x - mu) ** 2 * w).sum()


def load_daily_weights(date: pd.Timestamp) -> pd.Series:

    fname = f"weights{date.strftime('%Y%m%d')}.csv"
    fpath = os.path.join(WEIGHTS_DIR, fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Weight file not found: {fpath}")
    df = pd.read_csv(fpath, header=None, names=["ticker", "weight"], dtype=str)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.dropna(subset=["weight"])

    return df.set_index("ticker")["weight"].astype(float)


returns_df = pd.read_csv(
    os.path.join(DATA_DIR, "excess_returns.csv"), index_col=0, parse_dates=True
)
factor_loadings_df = pd.read_csv(
    os.path.join(DATA_DIR, "factor_loadings_df.csv"), header=[0, 1], index_col=0, parse_dates=True
)
factor_returns_df = pd.read_csv(
    os.path.join(DATA_DIR, "factor_returns.csv"), index_col=0, parse_dates=True
)
residuals_df = pd.read_csv(
    os.path.join(DATA_DIR, "regression_residuals.csv"), index_col=0, parse_dates=True
)

returns_df = returns_df.iloc[:-1]
factor_loadings_df = factor_loadings_df.iloc[:-1]
factor_returns_df = factor_returns_df.iloc[:-1]
residuals_df = residuals_df.iloc[:-1]

records = []
for dt in returns_df.index:

    r = returns_df.loc[dt].dropna()  # 实际收益
    res = residuals_df.loc[dt].reindex(r.index)  # 残差
    w = load_daily_weights(dt).reindex(r.index).fillna(0.0)

    cols = factor_loadings_df.columns
    overlap0 = len(set(cols.get_level_values(0)) & set(returns_df.columns))
    overlap1 = len(set(cols.get_level_values(1)) & set(returns_df.columns))
    factor_level = 0 if overlap1 > overlap0 else 1

    # (asset × factor) 暴露矩阵
    X = factor_loadings_df.loc[dt].unstack(level=factor_level)

    f = factor_returns_df.loc[dt]

    pred = X.dot(f)  # 因子预测收益

    tot_var = weighted_variance(r, w)
    res_var = weighted_variance(res, w)
    fac_var = weighted_variance(pred, w)

    records.append(
        {
            "date": dt,
            "residual_ratio": res_var / tot_var if tot_var else np.nan,
            "factor_ratio": fac_var / tot_var if tot_var else np.nan,
            "sum_ratio": (res_var + fac_var) / tot_var if tot_var else np.nan,
        }
    )

check_df = pd.DataFrame.from_records(records).set_index("date")

print("\n######  Variance attribution ratios  ######")
print(check_df.mean().rename("Mean over all timespan"))

out_path = f"{DATA_DIR}/attribution_ratios.csv"
check_df.to_csv(out_path, float_format="%.10f")
print(f"[DONE] saved {out_path}   shape={check_df.shape}")
