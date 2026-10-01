import pandas as pd
import numpy as np

def annualize_stats(mean, vol):
    """
    Annualize mean return and volatility.
    """
    ann_mean = mean * 12
    ann_vol = vol * np.sqrt(12)

    return ann_mean, ann_vol


def tangency_weights(returns, mean = None):
    """
    Calculate tangency portfolio weights.
    """
    if mean is None:
        mean = returns.mean()

    cov = returns.cov()
    inv_cov = np.linalg.inv(cov)

    weights = inv_cov @ mean
    weights = weights / weights.sum()

    return pd.Series(weights, index = returns.columns)


def portfolio_stats(returns, weights):
    """
    Calculate portfolio mean, volatility, and Sharpe ratio.
    """
    mean = returns.mean() @ weights
    vol = np.sqrt(weights.T @ returns.cov() @ weights)

    ann_mean, ann_vol = annualize_stats(mean, vol)

    sharpe = ann_mean / ann_vol

    return ann_mean, ann_vol, sharpe


def scale_weights_to_target(weights, mean_returns, target_return):
    """
    Scale portfolio weights to achieve a target expected return.
    """
    portfolio_mean = weights @ mean_returns
    scale = target_return / portfolio_mean

    return weights * scale