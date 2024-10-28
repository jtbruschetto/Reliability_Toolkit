from typing import Optional

import rainflow
import numpy as np
import pandas as pd

'''Time Series Analysis'''


def rain_flow(time_series: np.array|list|pd.DataFrame, span: Optional[int], binsize:int =1):
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
        df = pd.DataFrame(time_series)
    if span:
        df = df.ewm(span=span).mean()
    counts = rainflow.count_cycles(df, binsize=binsize)
    counts = pd.DataFrame(counts, columns=['range', 'counts'], dtype=float)
    counts.set_index('range', inplace=True, drop=True)
    return counts


def time_at_temp(df: pd.DataFrame):
    """

    :param df: Dataframe with columns [time, temp]
    :return: DataFrame of temp, dt_time
    """
    tdf = df.copy()
    tdf['temp'] = tdf['temp'].mul(2).round().div(2)
    df2 = tdf.groupby('temp')['dt_time'].sum().reset_index()
    df2.set_index('temp', inplace=True, drop=True)
    return df2
