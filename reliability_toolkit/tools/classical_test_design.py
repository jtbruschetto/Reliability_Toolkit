#global imports
import math
from typing import Optional
from scipy.stats import chi2
import numpy as np
import matplotlib.pyplot as plt
import re
#local imports
from reliability_toolkit.tools.utils import get_range_array

__all__ = ['ClassicalTestDesign', 'calculate_reliability', 'calculate_confidence', 'calculate_sample_size', 'calculate_life_ratio', 'calculate_allowable_failures']

class ClassicalTestDesign:
    def __init__(
        self,
        reliability: Optional[float] = None,
        confidence: Optional[float] = None,
        failures: Optional[int] = None,
        sample_size: Optional[int] = None,
        life_ratio: Optional[float] = None,
        beta: float = -1,
        **kwargs
):
        # Initialize Parameters
        """
        Initialize the ClassicalTestDesign object.

        This class allows you to determine key parameters for designing
        reliability tests based on Classical Test Theory (CTT).
        You can calculate one unknown parameter (reliability, confidence,
        failures, sample size, or life ratio) by providing the others.
        It also provides methods for updating parameters and visualizing
        the relationships between them.

        At least one of reliability, confidence, failures, sample_size, life_ratio must be None.
        The object will then calculate the missing parameter.

        :param reliability: The reliability target (0-1)
        :param confidence: The confidence interval (1 - alpha)
        :param failures: The allowed/observed number of failures
        :param sample_size: The number of samples on test
        :param life_ratio: The ratio of test time vs service life
        :param beta: The component Weibull shape parameter
        :param kwargs: keyword arguments passed to the calculation functions
        :raises ValueError: if too many arguments are provided
        """
        self.reliability = kwargs['reliability'] if 'reliability' in kwargs else reliability
        self.confidence = kwargs['confidence'] if 'confidence' in kwargs else confidence
        self.failures = kwargs['failures'] if 'failures' in kwargs else failures
        self.sample_size = kwargs['sample_size'] if 'sample_size' in kwargs else sample_size
        self.life_ratio = kwargs['life_ratio'] if 'life_ratio' in kwargs else life_ratio
        self.beta = kwargs['beta'] if 'beta' in kwargs else beta

        # Validate Arguments
        self.validate_inputs()

        # Check To Make sure 1 Argument is unknown for calculation
        if len([x for x in list(self.__dict__.values()) if x is None])>1:
            raise ValueError('Insufficient arguments provided, Too many arguments to solve for')

        # Solve for Unknown Argument
        if self.reliability is None:
            self.reliability = calculate_reliability(**self.__dict__)
        if self.confidence is None:
            self.confidence = calculate_confidence(**self.__dict__)
        if self.sample_size is None:
            self.sample_size = calculate_sample_size(**self.__dict__)
        if self.life_ratio is None:
            self.life_ratio = calculate_life_ratio(**self.__dict__)
        if self.failures is None:
            self.failures = calculate_allowable_failures(**self.__dict__)

    def get_results(self):
        return self.__dict__

    def validate_inputs(self):
        if self.reliability is not None and (self.reliability < 0 or self.reliability >1):
            raise ValueError('Reliability must be between a percentage between 0-1')
        if self.confidence is not None and (self.confidence < 0 or self.confidence >1):
            raise ValueError('confidence must be between a percentage between 0-1')
        if self.failures is not None and self.failures <0:
            raise ValueError('Failures must be a positive integer or 0')
        if self.sample_size is not None and self.sample_size <=0:
            raise ValueError('Sample Size must be a positive integer')
        if self.life_ratio is not None and self.life_ratio <=0:
            raise ValueError('Life Ratio must be a positive number')
        if self.beta <= 0 :
            raise ValueError('Beta is a required argument and must be a positive Number')

    def calculate_sample_size(self):
        return calculate_sample_size(**self.__dict__)

    def calculate_confidence(self):
        return calculate_confidence(**self.__dict__)

    def calculate_reliability(self):
        return calculate_reliability(**self.__dict__)

    def calculate_life_ratio(self):
        return calculate_life_ratio(**self.__dict__)

    def calculate_allowable_failures(self):
        return calculate_allowable_failures(**self.__dict__)

    def update_and_calculate_samples(self, **kwargs):
        self.__dict__.update(**kwargs)
        self.validate_inputs()
        return calculate_sample_size(**self.__dict__)

    def update_and_calculate_confidence(self, **kwargs):
        self.__dict__.update(**kwargs)
        self.validate_inputs()
        return calculate_confidence(**self.__dict__)

    def update_and_calculate_r_ts(self, **kwargs):
        self.__dict__.update(**kwargs)
        self.validate_inputs()
        return calculate_reliability(**self.__dict__)

    def update_and_calculate_life_ratio(self, **kwargs):
        self.__dict__.update(**kwargs)
        self.validate_inputs()
        return calculate_life_ratio(**self.__dict__)

    def update_and_calculate_failures(self, **kwargs):
        self.__dict__.update(**kwargs)
        self.validate_inputs()
        return calculate_allowable_failures(**self.__dict__)

    def plot_contour_by_func(self, x_axis, y_axis, contour_var,
                             fig: plt.Figure | None = None,
                             ax: plt.Axes | None = None,
                             cb=None, **kwargs):
        if (fig is not None) and (ax is not None):
            fig = fig
            ax = ax
        elif fig is not None:
            fig = fig
            ax = fig.add_subplot(111)
        elif ax is not None:
            fig = ax.figure
            ax = ax
        else:
            fig, ax = plt.subplots()

        if cb is not None:
            cb = cb

        # Show Target X Axis
        x_axis_target = self.__dict__[x_axis]
        ax.axvline(x=x_axis_target, color='k', linestyle='dashed', linewidth=1,
                   label=f'Target {x_axis.title()} = {x_axis_target:.2f}')
        x_axis_range = get_range_array(x_axis_target, x_axis)

        # Show List of Variables
        y_axis_target = self.__dict__[y_axis]
        ax.axhline(y=y_axis_target, color='k', linestyle='dashed', linewidth=1,
                   label=f'Target {y_axis.title()} = {y_axis_target:.2f}')
        y_axis_range = get_range_array( y_axis_target, y_axis)

        [X, Y] = np.meshgrid(x_axis_range, y_axis_range)
        if contour_var == 'sample_size':
            func = np.vectorize(self.update_and_calculate_samples)
            z_result = f'{self.update_and_calculate_samples(**{x_axis: x_axis_target, y_axis: y_axis_target}):.0f}'
            target = self.update_and_calculate_samples(**{x_axis: x_axis_target, y_axis: y_axis_target})
            levelsf = get_range_array(target, levels=100, min=False)
            levelsk = get_range_array(target, levels=10, min=False)
        elif contour_var == 'confidence':
            func = np.vectorize(self.update_and_calculate_confidence)
            z_result = f'{self.update_and_calculate_confidence(**{x_axis: x_axis_target, y_axis: y_axis_target}):.2f}'
            target = self.update_and_calculate_confidence(**{x_axis: x_axis_target, y_axis: y_axis_target})
            levelsf = get_range_array(target, 'confidence', levels=100)
            levelsk = get_range_array(target, 'confidence', levels=10)
        elif contour_var == 'reliability':
            func = np.vectorize(self.update_and_calculate_r_ts)
            z_result = f'{self.update_and_calculate_r_ts(**{x_axis: x_axis_target, y_axis: y_axis_target}):.2f}'
            target = self.update_and_calculate_r_ts(**{x_axis: x_axis_target, y_axis: y_axis_target})
            levelsf = get_range_array(target, 'r_ts', levels=100)
            levelsk = get_range_array(target, 'r_ts', levels=10)
        elif contour_var == 'life_ratio':
            func = np.vectorize(self.update_and_calculate_life_ratio)
            z_result = f'{self.update_and_calculate_life_ratio(**{x_axis: x_axis_target, y_axis: y_axis_target}):.2f}'
            target = self.update_and_calculate_life_ratio(**{x_axis: x_axis_target, y_axis: y_axis_target})
            levelsf = get_range_array(target, levels=100)
            levelsk = get_range_array(target, levels=10)
        else:  # contour_var == 'mean_shift':
            func = np.vectorize(self.update_and_calculate_failures)
            z_result = f'{self.update_and_calculate_failures(**{x_axis: x_axis_target, y_axis: y_axis_target}):.2f}'
            target = self.update_and_calculate_failures(**{x_axis: x_axis_target, y_axis: y_axis_target})
            levelsf = get_range_array(target, levels=100, min=False)
            levelsk = get_range_array(target, levels=10, min=False)

        Z = func(**{x_axis: X, y_axis: Y})
        csf = ax.contourf(X, Y, Z, vmin=levelsf[0], vmax=levelsf[-1], levels=levelsf, cmap='rainbow')
        cs = ax.contour(X, Y, Z, vmin=levelsf[0], vmax=levelsf[-1], levels=levelsk, colors='k')
        cb = fig.colorbar(csf, ax=ax)
        cb.ax.get_yaxis().labelpad = 15
        cb.ax.set_ylabel(contour_var.title(), rotation=270)
        ax.clabel(cs, inline=1, fontsize=10)
        ax.set_title(f'{re.sub(r"(?<=\w)([A-Z])", r" \1", self.__class__.__name__)}')
        ax.grid(True)
        ax.set_xlabel(f'{x_axis.title()}')
        ax.set_ylabel(f'{y_axis.title()}')
        ax.set_xlim(x_axis_range[0], x_axis_range[-1])
        ax.set_ylim(y_axis_range[0], y_axis_range[-1])
        ax.annotate(f'{contour_var.title()}: {z_result}', xy=(x_axis_target, y_axis_target),
                    xytext=(.99, .01), textcoords='axes fraction',
                    arrowprops=dict(arrowstyle="->"),
                    bbox={'boxstyle': "round", 'fc': "w"},
                    fontsize=7,
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    zorder=10
                    )
        ax.legend()
        return cb


def calculate_reliability(confidence: float, beta: float, failures: int, sample_size: int, life_ratio: float,
                          **_kwargs):

    """
    Calculate the Reliability achieved with a given test design
    :param confidence: Confidence interval (1 - alpha)
    :param beta: Component Weibull Shape parameter.
    :param failures: allowed/observed number of failures.
    :param sample_size: Number of samples on test.
    :param life_ratio: Ratio of test time vs service life
    :return: Reliability (0-1) as a percent
    """
    return math.exp(0 - ((chi2.ppf(confidence, df=(2 * failures + 2))) / (2 * sample_size * (life_ratio ** beta))))


def calculate_confidence(beta: float, failures: int, reliability: float, sample_size: int, life_ratio: float, **_kwargs):
    """
    Calculate the confidence achieved with a given test design
    :param beta: Component Weibull Shape parameter.
    :param failures: allowed/observed number of failures.
    :param reliability: Service Life Reliability Target.
    :param sample_size: Number of samples on test.
    :param life_ratio: Ratio of test time vs service life
    :return: Confidence ( 1 - Alpha )
    """
    return chi2.cdf(-2 * sample_size * life_ratio ** beta * math.log(reliability), df=(2 * failures + 2))


def calculate_sample_size(confidence: float, beta: float, failures: int, reliability: float, life_ratio: float, **_kwargs):
    """
    Calculate sample size needed to achieve Reliability target with the given test design
    :param confidence: Confidence interval (1 - alpha)
    :param beta: Component Weibull Shape parameter.
    :param failures: allowed/observed number of failures.
    :param reliability: Service Life Reliability Target.
    :param life_ratio: Ratio of test time vs service life
    :return: Sample size
    """
    return math.ceil((-chi2.ppf(confidence, df=(2 * failures + 2))) / 2 / (life_ratio ** beta) / math.log(reliability))


def calculate_life_ratio(confidence: float, beta: float, failures: int, reliability: float, sample_size: int, **_kwargs):
    """
    Calculate life ration needed to achieve Reliability target with the given test design
    :param confidence: Confidence interval (1 - alpha)
    :param beta: Component Weibull Shape parameter.
    :param failures: allowed/observed number of failures.
    :param reliability: Service Life Reliability Target.
    :param sample_size: Number of samples on test.
    :return: life ratio
    """
    return ((-chi2.ppf(confidence, df=(2 * failures + 2))) / 2 / sample_size / math.log(reliability)) ** (1 / beta)


def calculate_allowable_failures(confidence, beta, reliability, sample_size, life_ratio, **_kwargs):
    failures = 0
    calc_reliability = 1
    while calc_reliability >= reliability:
        failures += 1
        calc_reliability = calculate_reliability(confidence, beta, failures, sample_size, life_ratio)
    return failures - 1



if __name__ == "__main__":

    inputs = {
        # 'reliability': .95,
        'confidence': .6,
        'failures':0,
        'sample_size':12,
        'life_ratio':2,
        'beta':2.5,
    }
    # inputs.pop('beta')
    # inputs.pop('life_ratio')
    # inputs.update({'life_ratio': 0})
    # inputs.pop('sample_size')
    # inputs.update({'sample_size': 0})
    # inputs.pop('failures')
    # inputs.update({'failures': -1})
    # inputs.pop('confidence')
    # inputs.update({'confidence': 1.1})
    # inputs.pop('reliability')
    ctd = ClassicalTestDesign(**inputs)

    pass