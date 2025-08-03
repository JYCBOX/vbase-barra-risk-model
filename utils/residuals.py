import pandas as pd
from cross_section_regression import calculate_factor_returns_with_residuals

# === Load input data ===
weights_df = pd.read_csv("weights_df.csv", index_col=0, parse_dates=True)
excess_returns_df = pd.read_csv("excess_returns.csv", index_col=0, parse_dates=True)
factor_loadings_df = pd.read_csv(
    "factor_loadings_df.csv", header=[0, 1], index_col=0, parse_dates=True
)

# === Run cross-sectional regression (output factor returns + residuals ===
factor_returns_df, residuals_df = calculate_factor_returns_with_residuals(
    returns_df=excess_returns_df,
    exposures_df=factor_loadings_df,
    weights_df=weights_df,
)

# === Save results ===
factor_returns_df.to_csv("actor_returns.csv", float_format="%.10f")
residuals_df.to_csv("residuals.csv", float_format="%.10f")

print(f"[DONE] factor_returns shape = {factor_returns_df.shape}")
print(f"[DONE] residuals shape = {residuals_df.shape}")
