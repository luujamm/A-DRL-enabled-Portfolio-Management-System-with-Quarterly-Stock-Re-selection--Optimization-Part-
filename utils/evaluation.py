import numpy as np
from math import exp, log

EPS = 1e-8
TRADE_DAYS_PER_YEAR = 252
ANNUAL_RISK_FREE_RETURN = 1.01

def risk_free_return():
    day_return = exp(log(ANNUAL_RISK_FREE_RETURN) / TRADE_DAYS_PER_YEAR)
    return day_return - 1


def Sharpe_Ratio(daily_return):
    rfr = risk_free_return()
    SR = np.mean(daily_return - rfr + EPS) / (np.std(daily_return) + EPS) * TRADE_DAYS_PER_YEAR ** 0.5
    return SR


def Sortino_Ratio(daily_return):
    rfr = risk_free_return()
    negative_daily_return = daily_return[daily_return < 0]
    StR = np.mean(daily_return - rfr + EPS) / (np.std(negative_daily_return) + EPS) * TRADE_DAYS_PER_YEAR ** 0.5
    return StR


def max_drawdown(values):
    """ Max drawdown. See https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp """
    if len(values) > 0:
        peak = values.max()
        trough = values[values.argmax():].min()
        mdd1 = (trough - peak + EPS) / (peak + EPS)
        mdd2 = max_drawdown(values[: values.argmax()])
        mdd = min(mdd1, mdd2)
    else:
        mdd = 0
    return mdd


def evaluation_metrics(daily_return, values):
    return Sharpe_Ratio(daily_return), Sortino_Ratio(daily_return), max_drawdown(values)

