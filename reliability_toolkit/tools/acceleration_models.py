from dataclasses import dataclass
from typing import Optional

import math

__all__ = ['CoffinManson', 'Arrhenius', 'InversePowerLaw', 'Peck']

@dataclass
class CoffinManson:
    """
    Coffin Manson Acceleration Model

    The Acceleration equation is typically written as: AF = (dt_acc / dt_use) ** cm_exp

    :param cm_exp: Coffin Manson Exponent
    """
    cm_exp: float

    def __post_init__(self):
        self.model = 'Coffin Manson'

        # Validation
        if self.cm_exp <= 0:
            raise ValueError('Dataclass CoffinManson: cm_exp must be greater than 0')

    def acceleration_factor(self, dt_acc: float, dt_use: float) -> float:
        """
        Calculates the Acceleration Factor for a Coffin Manson Equation

        Coffin Manson Model is used for Temperature Cycling Tests

        the Acceleration equation is typically written as: AF = (dt_acc / dt_use) ** cm_exp

        if dt_use == 0: return 0

        "AF" is the acceleration factor (R2/R1)
        "dt_acc" is the temperature delta exercised in the accelerated test
        "dt_use" is the temperature delta seen in the field/use case
        "cm_exp" is the Coffin Manson Exponent

        :param dt_acc: Temperature Delta exercised in Accelerated Test
        :param dt_use: Temperature Delta seen in the Field/Use Case
        :return: Acceleration Factor
        """
        if dt_acc <= 0:
            raise ValueError('Dataclass CoffinManson: dt_acc must be greater than 0')
        if dt_use <= 0:
            raise ValueError('Dataclass CoffinManson: dt_use must be greater than 0')
        return (dt_acc / dt_use) ** self.cm_exp

    def test_cycles_conversion(self, cycles: dict[float, float], dt_eval: float) -> int:
        """
        Function to convert stress distribution into equivalent number of cycles at a given stress set point

        Example:
            - Field Stress Distribution to Equivalent Cycles at a single Test Stress
            - Variable Test Cycles to Equivalent Cycles at a single Test Stress

        Inputs:
        :param cycles: Dictionary of rainflow counted values [Temperature Delta as Key: Count of occurrences as Value]
        :param dt_eval: Desired Temperature Delta for Evaluation
        :return: Number of Test Cycles (int)
        """
        test_cycles = 0
        for dt_use, count in cycles.items():
            test_cycles += count / self.acceleration_factor(dt_eval, dt_use)
        return test_cycles

@dataclass
class Arrhenius:
    """
    Arrhenius Acceleration Model

    the Acceleration equation is typically written as: AF = exp((-Ea/k) (1/T_use - 1/T_acc))
    "Ea" is the activation energy (ev),

    :param ea: Activation Energy
    """
    ea: float  # Activation Energy
    t_rise: Optional[float] = None

    def __post_init__(self):
        self.model = 'Arrhenius'
        self.k = 8.617e-5 # Boltzmann constant
        self.kelvin = 273.15 # Conversion from Celsius to Kelvin

        # Validation
        if self.ea <= 0:
            raise ValueError('Dataclass Arrhenius: ea must be greater than 0')

    def rate_constant(self, temperature: float, a: float=1) -> float:
        """
        Calculates the rate constant for an Arrhenius Equation
        :param temperature: Reaction Temperature (°C)
        :param a: Pre-exponential factor
        :return: Rate Constant
        """
        if self.t_rise:
            temperature += self.t_rise
        return a * math.exp(self.ea/(self.k * (temperature + self.kelvin)))

    def equivalent_stress(self, sum_stress_function: float) -> float:
        if self.t_rise:
            return (-self.ea/self.k ) / math.log(sum_stress_function) - self.t_rise - self.kelvin
        else:
            return (-self.ea/self.k ) / math.log(sum_stress_function) - self.kelvin

    def acceleration_factor(self, t_acc: float, t_use: float) -> float:
        """
        Calculates the Acceleration Factor for an Arrhenius Equation
        https://www.itl.nist.gov/div898/handbook/apr/section1/apr151.htm

        the Acceleration equation is typically written as: AF = exp((-Ea/k) (1/T_use - 1/T_acc))
        "AF" is the acceleration factor (R2/R1)
        "Ea" is the activation energy (ev),
        "k" is Boltzmann's constant, 8.617e-5 (ev/K)
        "T_use" is the normal operating temperature (Converting °C to K),
        "T_acc" is the accelerated test temperature (Converting °C to K)

        :param t_acc: Test Temperature (°C)
        :param t_use: Field Temperature (°C)
        :return: Acceleration Factor
        """
        return math.exp((-self.ea/self.k) * ((1 / (t_acc + self.kelvin)) - (1 / (t_use + self.kelvin))))

    def test_hours_conversion(self, hours: dict[float, float], t_eval: float) -> float:
        """
        Function to convert stress distribution into equivalent number of hours at a given stress set point

        Example:
            - Field Stress Distribution to Equivalent Hours at a single Test Stress
            - Variable Test Hours to Equivalent Hours at a single Test Stress

        Inputs
        :param hours: Dictionary of time at temperature recorded values [Temperature as Key: Accumulated Time in Hours as Value]
        :param t_eval: Temperature in the Test
        :return: Number of Test Hours (int)
        """
        test_hours = 0
        for t_use, count in hours.items():
            test_hours += count / self.acceleration_factor(t_eval, t_use)
        return test_hours

@dataclass
class InversePowerLaw:
    n: float # Model Exponent

    def __post_init__(self):
        self.model = 'Inverse Power Law'

        # Validation
        if self.n <= 0:
            raise ValueError('Dataclass InversePowerLaw: n must be greater than 0')

    def acceleration_factor(self, s_acc: float, s_use: float) -> float:
        """
        Calculates the Acceleration Factor for an Inverse Power Law Equation

        the Acceleration equation is typically written as: AF = (S_use / S_acc) ** n
        "AF" is the acceleration factor (R2/R1)
        "S_acc" is the accelerated test stress
        "S_use" is the normal operating stress
        "n" is the model exponent

        :param s_acc: Accelerated Test S
        :param s_use: Normal Operating/Field Stress S
        :return: Acceleration Factor
        """
        if s_acc <= 0:
            raise ValueError('Dataclass InversePowerLaw: s_acc must be greater than 0')
        if s_use <= 0:
            raise ValueError('Dataclass InversePowerLaw: s_use must be greater than 0')
        return (s_use / s_acc) ** self.n

@dataclass
class Peck:
    n_exp: float # Relative Humidity Exponential Factor
    ea: float # Activation Energy

    def __post_init__(self):
        self.model = 'Peck'

        # Validation
        if self.n_exp <= 0:
            raise ValueError('Dataclass Peck: n_exp must be greater than 0')
        if self.ea <= 0:
            raise ValueError('Dataclass Peck: ea must be greater than 0')

    def acceleration_factor(self, t_use, t_acc, rh_use, rh_acc):
        """
        Temperature and Humidity Accelerated Life Model

        :model_param ea: Activation Energy (Ea)
        :model_param n_exp: Relative Humidity Exponential Factor

        :param t_use: Field Temperature (°C)
        :param t_acc: Test Temperature (°C)
        :param rh_use: Field Relative Humidity (%)
        :param rh_acc: Test Relative Humidity (%)
        :return: Acceleration Factor
        """
        if rh_use <= 0:
            raise ValueError('Dataclass Peck: rh_use must be greater than 0')
        if rh_acc <= 0:
            raise ValueError('Dataclass Peck: rh_acc must be greater than 0')
        return (rh_use/rh_acc)**(-self.n_exp) * math.exp((-self.ea/8.617e-5) * ((1 / (t_acc + 273.15)) - (1 / (t_use + 273.15))))



if __name__ == '__main__':
    test_dict = {
        5: 1000,
        10: 500,
        15: 100,
        20: 50,
        25: 25,
        30: 10,
        35: 5,
        40: 1,
        45: 1
    }


    cm = CoffinManson(cm_exp=2)
    print(f'Coffin Manson Acceleration Factor (AF): {cm.acceleration_factor(dt_acc=140, dt_use=100):.2f}')
    print(f'Test Cycles (Coffin Manson Model): {cm.test_cycles_conversion(test_dict, 60)} cycles')

    arr = Arrhenius(ea=.5)
    print(f'Arrhenius Acceleration Factor (AF): {arr.acceleration_factor(t_acc=110, t_use=80):.2f}')
    print(f'Test Hours (Arrhenius Model): {arr.test_hours_conversion(test_dict, 60):.2f} hours')