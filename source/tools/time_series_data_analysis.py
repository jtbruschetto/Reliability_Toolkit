from typing import Optional

import rainflow
import numpy as np
import numpy.typing as npt
import pandas as pd
import time

''' Timer Function Wrapper for Evaluation'''

def timer_wrapper(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        print(f'Time taken for {func.__name__}: {elapsed:.6f} seconds')
        return result
    return wrapper


'''Time Series Analysis'''

# @timer_wrapper
def rain_flow(time_series: npt.ArrayLike | list | pd.DataFrame, span: Optional[int] = None, binsize:int =1):
    """
    Input thermal time series data, and output table of counts per delta T range
    :param time_series: Time series temperature date for a single column
    :param span: Exponentially weighted mean range
    :param binsize: Delta T bin size
    :return: dataframe
    """
    if isinstance(time_series, pd.DataFrame):
        df = time_series.copy()
    else:
        df = pd.Series(time_series)
    if span:
        df = df.ewm(span=span).mean()
    counts = rainflow.count_cycles(df, binsize=binsize)
    counts = pd.DataFrame(counts, columns=['range', 'counts'], dtype=float)
    counts.set_index('range', inplace=True, drop=True)
    return counts

# @timer_wrapper
def time_at_temp(df: pd.DataFrame):
    """

    :param df: Dataframe with columns [time, temp]
    :return: DataFrame of temp, dt_time
    """
    tdf = df.copy()
    if len(tdf.columns) == 2:
        if (tdf.columns != ['time', 'temp']).all():
            tdf.columns = ['time', 'temp']
    else:
        ValueError('Input Dataframe must have 2 columns')
    tdf['dt_time'] = tdf['time'].diff()
    tdf['temp'] = tdf['temp'].mul(2).round().div(2)
    df2 = tdf.groupby('temp')['dt_time'].sum().reset_index()
    df2.set_index('temp', inplace=True, drop=True)
    return df2


