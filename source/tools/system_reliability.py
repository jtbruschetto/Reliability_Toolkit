import math

from reliability_dataclasses import *

def mean_time_between_failures(total_hours_operation: float, total_failures: int):
    return MeanTimeBetweenFailures(total_failures / total_hours_operation)


def series_reliability(reliability_values: list, **_kwargs):
    """
    Calculate System Reliability given a series system architecture and a list of Reliability values
    :param reliability_values: list of Reliability values (0-1) as a percents
    :return: Reliability (0-1) as a percent
    """
    return math.prod(reliability_values)

def parallel_reliability(reliability_values: list, **_kwargs):
    """
    Calculate System Reliability given a parallel system architecture and a list of Reliability values
    :param reliability_values: list of Reliability values (0-1) as a percents
    :return: Reliability (0-1) as a percent
    """
    return 1 - math.prod([1 - x for x in reliability_values])

def series_and_parallel_reliability(reliability_system: list, **_kwargs):
    """
    Calculate System Reliability given a list of series and parallel system architecture
    :param reliability_system: list of Reliability values (0-1) as a percents [[.9, .9, .8], [.9], [.9, [.9], [.8,.8]]]
    :return: Reliability (0-1) as a percent
    """
    series_items = [x for x in reliability_system if not isinstance(x, list)]
    print(series_items)
    parallel_items = [x for x in reliability_system if isinstance(x, list)]
    print(parallel_items)
    return series_reliability(series_items) * math.prod([parallel_reliability(x) for x in parallel_items])

if __name__ == '__main__':
    r = [.9, .9, .8]
    print(f'{r} | Series: {series_reliability(r)*100:.2f}% | Parallel: {parallel_reliability(r)*100:.2f}% ')
    r = [.9]
    print(f'{r} | Series: {series_reliability(r) * 100:.2f}% | Parallel: {parallel_reliability(r) * 100:.2f}% ')
    r = [.9, [.9], [.8,.8]]
    print(f'{r} | Series and Parallel: {series_and_parallel_reliability(r)*100:.2f}%')
    mttf = MeanTimeToFailure(10, 'Hours')
    print(mttf)