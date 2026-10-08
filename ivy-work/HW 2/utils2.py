import pandas as pd
import numpy as np
import statsmodels.api as sm

def max_drawdown_stats(returns):
    returns = returns.dropna()

    #calculate cumulative wealth and running maximum
    wealth = (1 + returns).cumprod()
    wealth = pd.concat([pd.Series([1.0], index = [returns.index[0] - pd.Timedelta(days = 1)]), wealth])

    running_max = wealth.cummax()

    drawdowns = wealth / running_max - 1

    trough_date = drawdowns.idxmin()
    peak_date = wealth.loc[:trough_date].idxmax()

    recovery = wealth.loc[trough_date:]
    recovery = recovery[recovery >= wealth.loc[peak_date]]
    recovery_date = recovery.index[0] if not recovery.empty else pd.NaT

    return pd.Series({"Max Drawdown": drawdowns.loc[trough_date],
                      "Peak Date": peak_date,
                      "Trough Date": trough_date,
                      "Recovery Date": recovery_date})


def regression_stats(returns, benchmark, rf = None, periods = 12):
    #align returns, benchmark, and optional risk-free rate
    data = pd.concat([returns.rename("asset"), benchmark.rename("benchmark")], axis = 1)

    if rf is not None:
        data["rf"] = rf

    data = data.dropna()

    #run regression with intercept
    y = data["asset"]
    x = sm.add_constant(data["benchmark"])

    model = sm.OLS(y, x).fit()

    #extract regression statistics
    alpha = model.params["const"]
    beta = model.params["benchmark"]
    residual_vol = model.resid.std()

    #calculate annualized statistics
    excess_return = y - data["rf"] if rf is not None else y

    return pd.Series({"Annualized Alpha": alpha * periods,
                      "Market Beta": beta,
                      "Treynor Ratio": excess_return.mean() * periods / beta if abs(beta) > 1e-12 else np.nan,
                      "Information Ratio": alpha / residual_vol * np.sqrt(periods) if residual_vol > 1e-12 else np.nan,
                      "R-squared": model.rsquared})