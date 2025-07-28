import pandas as pd

from .cross_section_regression import calculate_factor_returns

weights_df = pd.read_csv("../vbase-data/weights_df.csv", index_col=0, parse_dates=True)
excess_returns_df = pd.read_csv("../vbase-data/excess_returns.csv", index_col=0, parse_dates=True)
factor_loadings_df = pd.read_csv(
    "../vbase-data/factor_loadings_df.csv", header=[0, 1], index_col=0, parse_dates=True
)

factor_returns = calculate_factor_returns(
    returns_df=excess_returns_df,
    exposures_df=factor_loadings_df,
    weights_df=weights_df,
)

factor_returns.to_csv("../vbase-data/factor_returns.csv", float_format="%.10f")
print(f"[DONE] save as：factor_returns.csv  with shape = {factor_returns.shape}")
