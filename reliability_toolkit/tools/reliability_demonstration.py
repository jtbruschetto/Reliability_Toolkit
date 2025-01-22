import math
from scipy.stats import chi2

__all__ = [
    'demonstrated_reliability',
    'combined_demonstrated_reliability',
]

def demonstrated_reliability(confidence: float, beta=0., zbar=0., sample_size=0, life_ratio=1., failures=0, **_kwargs):
    """
    :param confidence: between 0 - 1
    :param beta: Weibull shape parameter
    :param zbar:
    :param sample_size: samples on test.
    :param life_ratio: test durations in number of lives completed
    :param failures: allowable/observed number of failures
    :return: demonstrated rel percent between 0 - 1
    """

    if life_ratio == 0:
        dem_rel = 0
    elif zbar > 0:
        numerator = chi2.ppf(confidence, df=(2 * failures + 2))  # 2 * acceleration_factor +
        denominator = 2 * zbar
        dem_rel = math.exp(0 - (numerator / denominator))
    else:
        numerator = chi2.ppf(confidence, df=(2 * failures + 2))  # 2 * acceleration_factor +
        denominator = 2 * int(sample_size) * float(life_ratio) ** float(beta)
        dem_rel = math.exp(0 - (numerator / denominator))
    " (-chi2.ppf(confidence, df=(2 * failures + 2))) / 2 / (life_ratio ** beta) / math.log(r_ts)"
    return dem_rel


def combined_demonstrated_reliability(z_bar_list, confidence=.60, failures=0):
    if sum(z_bar_list) == 0:
        return 0
    else:
        return math.exp(0 - (chi2.ppf(confidence, df=(2 * failures + 2)) / (2 * sum(z_bar_list))))