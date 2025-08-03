import pandas as pd
import numpy as np

# === Load factor variance attribution results ===
variance_attribution_df = pd.read_csv("variance_attribution.csv", index_col=0)

# === Step 1: Compute volatility from variance ===
variance_attribution_df["specific_vol"] = np.sqrt(variance_attribution_df["specific_var"])
variance_attribution_df["factor_vol"] = np.sqrt(variance_attribution_df["factor_var"])
variance_attribution_df["total_vol"] = np.sqrt(variance_attribution_df["total_var"])

# === Step 2: Save as assetrisk_{timestamp}.csv ===
timestamp = pd.Timestamp.today().strftime("%Y%m%d")
variance_attribution_df.index.name = "asset"
variance_attribution_df.reset_index().to_csv(
    f"assetrisk_{timestamp}.csv", index=False, float_format="%.10f"
)

print(f"[DONE] assetrisk_{timestamp}.csv saved.")
