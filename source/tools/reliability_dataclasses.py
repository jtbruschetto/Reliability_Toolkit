from typing import Literal
from dataclasses import dataclass

@dataclass
class MeanTimeBetweenFailures:
    mtbf: float

    def __post_init__(self):
        if self.mtbf <= 0:
            raise ValueError('MTBF must be greater than 0')
        self.failure_rate = 1 / self.mtbf
        self.FIT = 1_000_000_000 / self.mtbf

    def __str__(self):
        return f'MTBF: {self.mtbf} days'


@dataclass
class MeanTimeToFailure:
    """
    Dataclass for Mean Time To Failure (MTTF):
    mttf: float (Total Operating Time(time_interval)/ Total Failures)
    unit: Literal['Seconds','Minutes','Hours', 'Days', 'Weeks', 'Months', 'Years'] = 'Hours'
    """
    mttf: float
    time_interval: Literal['Seconds','Minutes','Hours', 'Days', 'Weeks', 'Months', 'Years'] = 'Hours'

    def __post_init__(self):
        if self.mttf <= 0:
            raise ValueError('MTTF must be greater than 0')

    def __str__(self):
        return f'MTTF: {self.mttf} {self.time_interval}'

    def failure_rate(self):
        return 1 / self.mttf


@dataclass
class Reliability:
    reliability: float

    def __post_init__(self):
        if self.reliability < 0 or self.reliability > 1:
            raise ValueError('R must be between 0 and 1')

    def __str__(self):
        return f'R: {self.reliability*100:.2f}%'

    def failure_probability(self):
        return 1 - self.reliability

    def failure_rate(self):
        return 1 - self.reliability


@dataclass
class ReliabilityCost:
    components_per_vehicle: int
    cost_per_component: float
    labor_cost_per_hour: float
    labor_hours_per_vehicle: float

    def __post_init__(self):
        if self.components_per_vehicle <= 1:
            raise ValueError('Dataclass ReliabilityCost: Components per vehicle must be greater than 1')

        if self.cost_per_component <= 0:
            raise ValueError('Dataclass ReliabilityCost: Cost per component must be greater than $0.00')

        if self.labor_cost_per_hour <= 0:
            raise ValueError('Dataclass ReliabilityCost: Labor cost per hour must be greater than 0hrs')

        if self.labor_hours_per_vehicle <= 0:
            raise ValueError('Dataclass ReliabilityCost: Labor hours per vehicle must be greater than $0.00')

        self.repair_cost_per_vehicle = (self.components_per_vehicle * self.cost_per_component) + (self.labor_cost_per_hour * self.labor_hours_per_vehicle)


@dataclass
class WeibullInputs:
    beta: float
    eta: float
    gamma: float = 0

    def __post_init__(self):
        if self.eta <= 0:
            raise ValueError('Dataclass WeibullInputs: Eta must be greater than 0')

        if self.beta <= 0:
            raise ValueError('Dataclass WeibullInputs: Beta must be greater than 0')

        if self.gamma <= 0:
            raise ValueError('Dataclass WeibullInputs: gamma must be greater than 0')

    def reliability(self, t):
        from math import exp
        '''
        Caculates Reliability based on weibul Parameters
        t: float, eta: float, beta: float, gamma: float
        '''
        if t <= 0: return 1
        return exp(-((t - self.gamma) / self.eta) ** self.beta)

@dataclass
class VehicleProduction:
