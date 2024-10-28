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

