import numpy as np


def get_range_array( target, axis: str | None = None, levels=100, min=True):
    sensitivity = {
        0.99: np.linspace(.97, 1, levels, endpoint=False),
        0.95: np.linspace(.9, 1, levels, endpoint=False),
        0.9: np.linspace(.8, 1, levels, endpoint=False),
        0.8: np.linspace(.7, 1, levels, endpoint=False),
        0.7: np.linspace(.6, 1, levels, endpoint=False),
        0.6: np.linspace(.5, 1, levels, endpoint=False),
    }
    if axis in ['confidence', 'r_ts']:
        if target >= 0.99:
            axis_range = sensitivity[0.99]
        elif target >= 0.95:
            axis_range = sensitivity[0.95]
        elif np.round(target, 1) in sensitivity:
            axis_range = sensitivity[np.round(target, 1)]
        else:
            axis_range = np.linspace(0, 1, levels, endpoint=False)
    else:
        if min:
            axis_range = np.linspace(target * .5, target * 2, levels)
        else:
            axis_range = np.linspace(0, (target//5 + 1)*5 * 2, levels, endpoint=False)
    return axis_range

def seconds_to_minutes(seconds):
    return seconds / 60

def seconds_to_hours(seconds):
    return seconds / 3600

def minutes_to_hours(minutes):
    return minutes / 60

def minutes_to_seconds(minutes):
    return minutes * 60

def hours_to_minutes(hours):
    return hours * 60

def hours_to_seconds(hours):
    return hours * 3600