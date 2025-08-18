import pandas as pd
import numpy as np

def compute_ewma_covariance(returns: pd.DataFrame, half_life_vol: int = 42, half_life_corr: int = 200) -> pd.DataFrame:
    """
    Compute EWMA covariance matrix with separate half-lives for volatility and correlation.

    Parameters
    ----------
    returns : pd.DataFrame
        Factor return time series. Rows = dates, columns = factor names.
    half_life_vol : int
        Half-life for exponential weighting of variance.
    half_life_corr : int
        Half-life for exponential weighting of correlation.

    Returns
    -------
    cov_matrix : pd.DataFrame
        Covariance matrix of factors using EWMA.
    """
    # === Step 1: Compute EWMA variance for each factor ===
    # Convert half-life to smoothing parameter lambda for volatility
    lambda_vol = 1 - np.log(2) / half_life_vol
    # Apply EWMA to compute time-varying variance; take the most recent estimate
    ewma_var = returns.ewm(alpha=1 - lambda_vol, adjust=False).var().iloc[-1]

    # === Step 2: Normalize returns by EWMA volatility to get standardized returns ===
    # This prepares the data for computing correlations separately
    standardized = returns / np.sqrt(ewma_var)

    # === Step 3: Compute EWMA correlation using standardized returns ===
    # Use a different (typically slower) half-life for correlation smoothing
    lambda_corr = 1 - np.log(2) / half_life_corr
    # Apply EWMA correlation with the new lambda; take the most recent correlation matrix
    ewma_corr = standardized.ewm(alpha=1 - lambda_corr, adjust=False).corr(pairwise=True).iloc[-1]

    # === Step 4: Reconstruct covariance matrix ===
    # cov(X, Y) = corr(X, Y) × std(X) × std(Y)
    cov_matrix = ewma_corr.values * np.outer(np.sqrt(ewma_var.values), np.sqrt(ewma_var.values))
    
    # Return covariance matrix as a DataFrame with factor names as index/columns
    return pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)


# === Load factor returns from CSV ===
factor_returns_df = pd.read_csv("factor_returns.csv", index_col=0, parse_dates=True)

# === Compute the EWMA-based factor covariance matrix using Barra USE4D settings ===
factor_cov = compute_ewma_covariance(factor_returns_df, half_life_vol=42, half_life_corr=200)

# === Save the result to CSV ===
factor_cov.to_csv("factor_covariance_matrix.csv", float_format="%.10f")
print("[DONE] factor_covariance_matrix.csv saved.")
