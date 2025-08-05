import pandas as pd
import numpy as np
from pathlib import Path

def load_outputs(dir_path: Path):
    fr = pd.read_csv(dir_path / "factor_returns.csv", index_col=0, parse_dates=True)
    rs = pd.read_csv(dir_path / "regression_residuals.csv", index_col=0, parse_dates=True)
    fl = pd.read_csv(dir_path / "factor_loadings.csv", header=[0,1], index_col=0, parse_dates=True)
    return fr, rs, fl

def average_exposure(fl):
    avg = fl.mean(axis=0)
    return avg.unstack(level=0)

def covariance_attribution(fr, rs, avg_exp):
    Σ = fr.cov()
    spec_var = rs.var(axis=0)
    assets = avg_exp.index
    fvar = {
        asset: avg_exp.loc[asset].reindex(Σ.columns, fill_value=0).values
               @ Σ.values
               @ avg_exp.loc[asset].reindex(Σ.columns, fill_value=0).values
        for asset in assets
    }
    df = pd.DataFrame({
        "factor_var": pd.Series(fvar),
        "specific_var": spec_var.reindex(assets)
    })
    df["total_var"] = df.factor_var + df.specific_var
    df["factor_share"] = df.factor_var / df.total_var
    df["spec_share"] = df.specific_var / df.total_var
    return df

def main():
    base = Path(".")
    fr, rs, fl = load_outputs(base)
    avg_exp = average_exposure(fl)
    attribution = covariance_attribution(fr, rs, avg_exp)
    out = base / "final_variance_attribution.csv"
    attribution.to_csv(out, float_format="%.6f")
    print(f"Saved attribution → {out}")

if __name__ == "__main__":
    main()```
