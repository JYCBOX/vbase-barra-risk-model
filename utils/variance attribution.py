import pandas as pd
import numpy as np

# === Load factor returns and residuals ===
factor_returns_df = pd.read_csv("factor_returns.csv", index_col=0, parse_dates=True)
residuals_df = pd.read_csv("residuals.csv", index_col=0, parse_dates=True)

# === Load factor loadings ===
factor_loadings_df = pd.read_csv(
    "factor_loadings_df.csv", header=[0, 1], index_col=0, parse_dates=True
)
# === Step 1: Compute factor covariance matrix ===
factor_cov = factor_returns_df.cov()  # (K, K)

# === Step 2: Compute residual variance (specific risk) for each asset ===
specific_var = residuals_df.var()  # (N_assets,)

# === Step 3: Compute average exposure vector for each asset ===
factor_names = factor_loadings_df.columns.get_level_values(0).unique()
asset_names = factor_loadings_df.columns.get_level_values(1).unique()

# Each asset's average exposure to each factor
# Note: factor_loadings_df.columns is a MultiIndex (level 0 = factor, level 1 = asset)
# Mean over time index → average exposure for each (factor, asset)
# .unstack(level=0) → index = asset, columns = factor
avg_exposure_df = (
    factor_loadings_df
    .mean(axis=0)               # average exposure for each (factor, asset)
    .unstack(level=0)           # index = asset, columns = factor
)

sector_factors = [
    'Communication Services', 'Consumer Discretionary', 'Consumer Staples',
    'Energy', 'Financials', 'Health Care', 'Industrials',
    'Information Technology', 'Materials', 'Real Estate', 'Utilities'
]

print(avg_exposure_df[sector_factors].describe().T)

# === Step 4: Factor risk = X Σ X^T ===
factor_var = avg_exposure_df.apply(
    lambda x: x.reindex(factor_cov.index).fillna(0).T @ factor_cov @ x.reindex(factor_cov.index).fillna(0),
    axis=1
)# index = asset

# === Step 5: Total risk = factor + specific (align index) ===
specific_var_aligned = specific_var.reindex(factor_var.index)
total_var = factor_var + specific_var_aligned

# === Step 6: Output attribution dataframe ===
variance_attribution_df = pd.DataFrame({
    "factor_var": factor_var,
    "specific_var": specific_var_aligned,
    "total_var": total_var,
    "factor_risk_pct": factor_var / total_var,
})


print("factor_var index:", factor_var.index[:5])
print("specific_var index:", specific_var.index[:5])
print("Index match check：", factor_var.index.equals(specific_var.index))

variance_attribution_df.to_csv("variance_attribution.csv", float_format="%.10f")
print("[DONE] variance_attribution.csv saved.")

# === Validation Tests ===

# Test 1: Check how many assets have specific_var > total_var (should be 0)
check1 = (variance_attribution_df["specific_var"] > variance_attribution_df["total_var"]).sum()
print("Number of assets with specific_var > total_var:", check1)

# Test 2: Check whether total variance ≈ factor + specific variance
diff = np.abs(
    variance_attribution_df["total_var"]
    - (variance_attribution_df["factor_var"] + variance_attribution_df["specific_var"])
)

max_diff = diff.max()
print("Max difference between total variance and factor + specific:", max_diff)

# Test 3: Check residual volatility < total volatility
resid_vol = np.sqrt(variance_attribution_df["specific_var"])
total_vol = np.sqrt(variance_attribution_df["total_var"])
check2 = (resid_vol > total_vol).sum()
print("Number of assets with residual risk > total risk:", check2)

# === Plot: Distribution of factor risk contribution ratio ===
import matplotlib.pyplot as plt

plt.hist(variance_attribution_df["factor_risk_pct"], bins=30, edgecolor="k")
plt.title("Distribution of Factor Risk Contribution (%)")
plt.xlabel("Factor Risk / Total Risk")
plt.ylabel("Asset Count")
plt.grid(True)
plt.show()
