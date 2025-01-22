from reliability_toolkit.tools.acceleration_models import CoffinManson, Arrhenius, Peck, InversePowerLaw
from dataclasses import dataclass

import math
from scipy.special import ndtri
from scipy.stats.distributions import lognorm
import pandas as pd
import matplotlib.pyplot as plt

__all__ = [
    'UsageDistribution',
    'StressDistribution',
    'EquivalentStressAndUsage',
    'calculate_reliability_from_usage_change',
    'create_usage_distribution',
    'calculate_usage_percentile_from_distribution',
    'create_generic_usage_distribution_lognorm',
    'calculate_equivalent_usage',
]

@dataclass
class UsageDistribution:
    yearly_usage_levels: list[float]
    usage_percentile: list[float]

    def __post_init__(self):
        self.df = pd.DataFrame(data={'yearly_usage_levels' : self.yearly_usage_levels, 'percentile': self.usage_percentile})

@dataclass
class StressDistribution:
    stress_levels: list[float]
    stress_percentile: list[float]

    def __post_init__(self):
        self.df = pd.DataFrame(data={'stress_levels' : self.stress_levels, 'percentile': self.stress_percentile})

@dataclass
class EquivalentStressAndUsage:
    beta: float # Component Weibull Shape parameter
    input_reliability: float # Reliability at Input Usage Stress (0-1)
    service_life: int # Service Life (years)
    damage_model: CoffinManson | Arrhenius | Peck | InversePowerLaw # Acceleration Model
    usage_distribution: UsageDistribution # Usage Distribution
    stress_distribution: StressDistribution # Stress Distribution

    def __post_init__(self):
        # validation
        if self.beta <= 0:
            raise ValueError('Beta must be greater than 0')
        if (self.input_reliability < 0) | (self.input_reliability > 1):
            raise ValueError('Reliability must be between a percentage between 0-1')
        if self.service_life <= 0:
            raise ValueError('Service Life must be greater than 0')

        self.process_usage_distribution()
        self.process_stress_distribution()

        print(self.stress_distribution.df)
        print(self.equivalent_stress)
        print(self.usage_distribution.df)
        print(self.equivalent_usage)

    def process_usage_distribution(self):
        """
        Calculates
            life_usage_levels: life usage levels
            pxu^B: percentile * life_usage_levels^beta
            equivalent_usage: equivalent usage
        """
        self.usage_distribution.df['life_usage_levels'] = self.usage_distribution.df.yearly_usage_levels * self.service_life
        self.usage_distribution.df['pxu^B'] = self.usage_distribution.df.percentile * self.usage_distribution.df.life_usage_levels ** self.beta
        self.equivalent_usage = self.usage_distribution.df['pxu^B'].sum() ** (1 / self.beta)

    def process_stress_distribution(self):
        """
        Calculates
            g_prime: function of stress condition (Arrhenius Rate Constant) with known parameters
            Percentile/g_prime: ratio of percentile stress and stress function
            S_eq: equivalent stress
        """
        self.stress_distribution.df['g_prime'] = self.stress_distribution.df.stress_levels.apply(self.damage_model.rate_constant)
        self.stress_distribution.df['Percentile/g_prime'] = self.stress_distribution.df.percentile / self.stress_distribution.df.g_prime
        self.equivalent_stress = self.damage_model.equivalent_stress(sum_stress_function=self.stress_distribution.df['Percentile/g_prime'].sum())

    def calculate_reliability_from_usage_change(self):
        pass

def calculate_reliability_from_usage_change(beta: float, input_reliability: float, input_usage: float, target_usage: float):
    """
    This function calculates the reliability at a target usage level, given the reliability at a different input
    usage level, using the Weibull distribution. It takes in the initial reliability, Weibull shape parameter (beta),
    input usage, and target usage, and returns the reliability at the target usage. The function also checks if the
    input reliability is within the valid range of 0 to 1.

    :param beta: Component Weibull Shape parameter.
    :param input_reliability: Reliability at Input Usage Stress (0-1)
    :param input_usage: Input Usage achieving reliability
    :param target_usage: Usage level for Reliability conversion
    :return reliability at target usage (0-1)
    """
    if beta <= 0:
        raise ValueError('Beta must be greater than 0')
    if (input_reliability < 0) | (input_reliability > 1):
        raise ValueError('Reliability must be between a percentage between 0-1')
    if input_usage <= 0:
        raise ValueError('Input Usage must be greater than 0')
    if target_usage <= 0:
        raise ValueError('Target Usage must be greater than 0')

    return math.exp(math.log(input_reliability) * (target_usage / input_usage) ** beta)


def create_usage_distribution(usage_distribution: dict[float: float]):
    """
    This function creates a usage distribution from a dictionary where the keys are the usage levels and the values are
    the corresponding counts of occurrences.

    :param usage_distribution: A dictionary where the keys are the usage levels and the values are the corresponding
                               counts.
    :return: A dictionary where the keys are the usage levels and the values are the corresponding probabilities.
    """

    return usage_distribution

def calculate_usage_percentile_from_distribution(usage_distribution: dict[float: float], percentile: float):
    """
    This function calculates the percentile of a given usage distribution. It takes in a dictionary where the keys are the
    usage levels and the values are the corresponding counts. It then returns the usage value at the given
    percentile.

    :param usage_distribution: A dictionary where the keys are the usage levels and the values are the corresponding
                               probabilities.
    :param percentile: The percentile of the usage distribution to be calculated
    :return: The usage value at the given percentile
    """


def create_generic_usage_distribution_lognorm(p: float, sigma=0.0, x_p=0.0, x_ave=0.0, x_median=0.0, x_e=200000,
                                              step=100, mode='Percentile', show_plot=False):
    """
    The create_generic_usage_distribution_lognorm function creates a generic usage distribution for the lognormal
    distribution. The function takes in the following parameters:

    :param p: float: Set the percentile of the distribution
    :param sigma: float: Set the standard deviation of the distribution
    :param x_p: Set the percentile value
    :param x_ave: Set the average value of the distribution
    :param x_median: Set the median value of the distribution
    :param x_e: int: Set the end of the x-axis for plotting
    :param step: int: Set the step size of the x-axis for plotting
    :param mode: Specify the type of distribution to be created
        Options:
            'Percentile'            requires: sigma, x_p, p
            'Average'               requires: sigma, x_ave
            'Median'                requires: sigma, x_median
            'Median&Percentile'     requires: x_median, x_p, p
    :param show_plot: Show the plot of the pdf and cdf of the distribution
    :return: A dictionary
    """
    if mode == 'Percentile':
        mu = math.log(x_p) - sigma * ndtri(p)
        x_ave = math.exp(mu + (sigma ** 2) / 2)
        x_median = math.exp(mu)
    elif mode == 'Average':
        mu = math.log(x_ave) - (sigma ** 2) / 2
        x_median = math.exp(mu)
    elif mode == 'Median':
        mu = math.log(x_median)
        x_ave = math.exp(mu + (sigma ** 2) / 2)
    elif mode == 'Median&Percentile':
        mu = math.log(x_median)
        sigma = (math.log(x_p)-mu)/ndtri(p)
        x_ave = math.exp(mu + (sigma ** 2) / 2)
    else:
        return

    x = range(0, x_e, step)
    df = pd.DataFrame(x, columns=['Usage'])
    df['PDF'] = lognorm.pdf(x=df['Usage'], s=sigma, scale=math.exp(mu))
    df['CDF'] = lognorm.cdf(x=df['Usage'], s=sigma, scale=math.exp(mu))

    if show_plot:
        df.plot(x='Usage', y=['PDF', 'CDF'], subplots=True)
        plt.show()

    results = {
        'x_e': x_e,
        'p': p,
        'sigma': sigma,
        'mu': mu,
        'x_p': x_p,
        'x_ave': x_ave,
        'x_median': x_median,
        'mode': mode,
        'dist': df
    }
    return results


def calculate_equivalent_usage(u_v: list, u_p: list, beta: float):
    return sum(u_pi * (u_vi ** beta) for u_pi, u_vi in zip(u_p, u_v)) ** (1 / beta)


if __name__ == '__main__':
    print(calculate_reliability_from_usage_change(2.5, .9, 150_000, 200_000))

    sd = StressDistribution(
        stress_levels=[
            -25,
            -20,
            -15,
            -10,
            -5,
            0,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            40,
            45,
            50,
        ],
        stress_percentile=[
            0.20 /100,
            0.50 /100,
            1.30 /100,
            2.00 /100,
            4.00 /100,
            6.00 /100,
            8.00 /100,
            10.00 /100,
            12.00 /100,
            13.00 /100,
            13.50 /100,
            12.00 /100,
            9.50 /100,
            6.00 /100,
            2.00 /100,
            0.00 /100,
        ]
    )
    ud = UsageDistribution(
        yearly_usage_levels=[
            100,
            200,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1_000,
            1_100,
            1_200,
            1_300,
            1_400,
            1_500,
            1_600,
            1_700,
            1_800,
            1_900,
            2_000,
            2_100,
            2_200,
            2_300,
            2_400,
            2_500,
        ],
        usage_percentile=[
            0.10 /100,
            0.50 /100,
            2.00 /100,
            6.00 /100,
            9.00 /100,
            11.00 /100,
            12.00 /100,
            11.00 /100,
            9.00 /100,
            7.50 /100,
            6.00 /100,
            5.00 /100,
            4.00 /100,
            3.50 /100,
            3.00 /100,
            2.50 /100,
            2.00 /100,
            1.70 /100,
            1.30 /100,
            1.00 /100,
            0.80 /100,
            0.60 /100,
            0.40 /100,
            0.20 /100,
            0.40 /100,
        ]
    )

    eqs = EquivalentStressAndUsage(
        beta=2,
        input_reliability=.98,
        service_life=10,
        damage_model=Arrhenius(
            ea=.7,
            t_rise=20
        ),
        usage_distribution=ud,
        stress_distribution=sd
    )


