import pandas as pd
import numpy as np
import statsmodels.api as sm

def robust_cross_section(returns, exposures, weights, tuning_const=1.345):
    y = returns.astype(float)
    X = exposures.astype(float)
    w = weights.astype(float)
    sw = np.sqrt(w)
    model = sm.RLM(endog=y * sw,
                   exog=X.mul(sw, axis=0),
                   M=sm.robust.norms.HuberT(t=tuning_const))
    res = model.fit()
    return pd.Series(res.params, index=X.columns)

def compute_factor_returns(returns_df, loadings_df, weights_df, tuning_const=1.345):
    returns_df.index = pd.to_datetime(returns_df.index)
    loadings_df.index = pd.to_datetime(loadings_df.index)
    weights_df.index = pd.to_datetime(weights_df.index)
    dates = returns_df.index.intersection(loadings_df.index).intersection(weights_df.index)
    fr_list = []
    for date in dates:
        y = returns_df.loc[date].dropna()
        X = loadings_df.loc[date]
        w = weights_df.loc[date].dropna()
        common = y.index.intersection(X.index).intersection(w.index)
        if not len(common):
            fr_list.append(pd.Series(np.nan, index=X.columns, name=date))
            continue
        fr = robust_cross_section(y.loc[common],
                                  X.loc[common],
                                  w.loc[common],
                                  tuning_const)
        fr.name = date
        fr_list.append(fr)
    return pd.DataFrame(fr_list)
