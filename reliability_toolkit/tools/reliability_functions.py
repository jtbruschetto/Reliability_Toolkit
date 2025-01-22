import math
import numpy as np
from scipy.stats import chi2

from rich import print

__all__ = [
    "rel_dictionary",
    "z_bar",
    "r_l",
    "r_u",
    "r_p",
    "n_field"]


def rel_dictionary(confidence: float, beta: float, zbar: float, failures: int, service_years: int, r_g: float,
                   **_kwargs):
    """
    The rel_dictionary function takes in the following arguments:
        confidence: The desired confidence level for the rel estimate.
        beta: The shape parameter of the Weibull distribution.
        zbar: The mean life of a sample from a population with known failure rate r_g.  This is usually calculated using
            an accelerated test, but can also be estimated from field data if available (see MIL-HDBK-189).
            If no value is provided, it will default to 1 year (365 days).

    :param confidence: float: Determine the confidence level of the rel estimate
    :param beta: float: Weibull shape parameter
    :param zbar: float: Calculate the upper and lower bounds of rel
    :param failures: int: Determine the number of failures in the field
    :param service_years: int: Calculate the n_l, n_u and n_p values
    :param r_g: float: Set the goal rel
    :return: A dictionary with the following keys:
    """
    rel_dict = {
        "r_l": r_l(failures, confidence, zbar),
        "r_u": r_u(failures, confidence, zbar),
        "r_p": r_p(failures, zbar),
        "zbar": zbar
    }
    rel_dict['n_l'] = n_field(service_years, rel_dict["r_l"], beta)
    rel_dict['n_u'] = n_field(service_years, rel_dict["r_u"], beta)
    rel_dict['n_p'] = n_field(service_years, rel_dict["r_p"], beta)
    # print(rel_dict)

    if rel_dict["r_l"] >= r_g:
        rel_dict['status'] = 'pass'
    elif rel_dict["r_u"] < r_g:
        rel_dict['status'] = 'Fail'
    else:
        rel_dict['status'] = 'Inconclusive'
    rel_dict['r_unproven'] = r_g - rel_dict["r_l"]

    return rel_dict

def z_bar(t, ts, beta, **_kwargs):
    """
    Calculate Z_bar
    :param t: Test time
    :param ts: Service life
    :param beta: Component Weibull Shape parameter.
    :return: z_bar
    """
    return (t / ts) ** beta

def r_l(failures: int, confidence: float, zbar: float):
    """
    The r_l function calculates the lower confidence bound for rel.

    :param failures: int: Determine the number of failures that have occurred
    :param confidence: float: Determine the confidence interval
    :param zbar: float: Calculate the r_l value
    :return: The lower bound of the confidence interval for rel
    """
    r = np.nan if zbar == 0 else math.exp(0 - (chi2.ppf(confidence, df=(2 * failures + 2)) / (2 * zbar)))
    return r

def r_u(failures: int, confidence: float, zbar: float):
    """
    The r_u function calculates the upper confidence bound for rel.

    :param failures: int: Calculate the critical value of chi-squared
    :param confidence: float: Determine the confidence level
    :param zbar: float: Calculate the r_u value
    :return: The upper bound of the confidence interval for rel
    """
    r = np.nan if zbar == 0 else math.exp(0 - (chi2.ppf(1 - confidence, df=(2 * failures)) / (2 * zbar)))
    return r

def r_p(failures: int, zbar: float):
    """
    The r_p function calculates the probability of failure for a given number of failures and zbar.

    :param failures: int: the number of failures observed
    :param zbar: float: Calculate the r_p value
    :return: The point estimate of the confidence interval for rel
    """
    r = np.nan if zbar == 0 else math.exp(0 - (failures / zbar))
    return r

def n_field(service_yrs: int, r: float, beta: float):

    """

    :param service_yrs: int: Specify the number of years in service
    :param r: float: Define the discount rate
    :param beta: float: Calculate the n value
    :return: The number of years until the field is depleted
    :doc-author: Trelent
    """
    n = np.nan if (r == 1) | (r == 0) else service_yrs * (-1 / math.log(r)) ** (1 / beta)
    return n


if __name__ == "__main__":
    print(rel_dictionary(.6, 2.5, 365, 1,10, .98))
    print(rel_dictionary(.6, 2.5, 100, 1, 10, .98))
    print(rel_dictionary(.6, 2.5, 20, 1, 10, .98))