import math

def coffin_manson_acceleration_factor(dt_acc: float, dt_use: float, cm_exp: float) ->float:
    """
    Calculates the Acceleration Factor for a Coffin Manson Equation

    Coffin Manson Model is used for Temperature Cycling Tests

    the Acceleration equation is typically written as: AF = (dt_acc / dt_use) ** cm_exp
    "AF" is the acceleration factor (R2/R1)
    "dt_acc" is the temperature delta exercised in the accelerated test
    "dt_use" is the temperature delta seen in the field/use case
    "cm_exp" is the Coffin Manson Exponent

    :param dt_acc: Temperature Delta exercised in Accelerated Test
    :param dt_use: Temperature Delta seen in the Field/Use Case
    :param cm_exp: Coffin Manson Exponent
    :return: Acceleration Factor
    """
    return (dt_acc / dt_use) ** cm_exp if dt_use != 0 else 0

def coffin_manson_cycles_conversion(cycles:dict[float, float], dt_eval:float, cm_exp:float) -> int:
    """
    Function to convert stress distribution into equivalent number of cycles at a given stress set point

    Example:
        - Field Stress Distribution to Equivalent Cycles at a single Test Stress
        - Variable Test Cycles to Equivalent Cycles at a single Test Stress

    Inputs:
    :param cycles: Dictionary of rainflow counted values [Temperature Delta as Key: Count of occurrences as Value]
    :param dt_eval: Desired Temperature Delta for Evaluation
    :param cm_exp: Coffin Manson Exponent
    :return: Number of Test Cycles (int)
    """
    test_cycles = 0
    for dt_use, count in cycles.items():
        test_cycles += count / coffin_manson_acceleration_factor(dt_eval, dt_use, cm_exp)
    return test_cycles

def arrhenius_acceleration_factor(ea: float, t_acc: float, t_use: float) -> float:
    """
    Calculates the Acceleration Factor for an Arrhenius Equation
    https://www.itl.nist.gov/div898/handbook/apr/section1/apr151.htm

    the Acceleration equation is typically written as: AF = exp((-Ea/k) (1/T_use - 1/T_acc))
    "AF" is the acceleration factor (R2/R1)
    "Ea" is the activation energy (ev),
    "k" is Boltzmann's constant, 8.617e-5 (ev/K)
    "T_use" is the normal operating temperature (Converting °C to K),
    "T_acc" is the accelerated test temperature (Converting °C to K)

    :param ea: Activation Energy (Ea)
    :param t_acc: Test Temperature (°C)
    :param t_use: Field Temperature (°C)
    :return: Acceleration Factor
    """
    return math.exp((-ea/8.617e-5) * ((1 / (t_acc + 273.15)) - (1 / (t_use + 273.15))))

def arrhenius_hours_conversion(hours:dict[float,float], t_eval: float, ea:float) -> float:

    """
    Function to convert field temperature distribution into test hours

    Example:
        - Field Stress Distribution to Equivalent Hours at a single Test Stress
        - Variable Test Hours to Equivalent Hours at a single Test Stress

    Inputs
    :param hours: Dictionary of time at temperature recorded values [Temperature as Key: Accumulated Time in Hours as Value]
    :param t_eval: Temperature in the Test
    :param ea: Activation Energy
    :return: Number of Test Hours (int)
    """
    test_hours = 0
    for t_use, count in hours.items():
        test_hours += count / arrhenius_acceleration_factor(ea, t_eval, t_use)
    return test_hours

def inverse_power_law_acceleration_factor(s_acc: float, s_use: float, n: float):
    """
    Calculates the Acceleration Factor for an Inverse Power Law Equation

    the Acceleration equation is typically written as: AF = (S_use / S_acc) ** n
    "AF" is the acceleration factor (R2/R1)
    "S_acc" is the accelerated test stress
    "S_use" is the normal operating stress
    "n" is the model exponent

    The parameter "n" in the inverse power relationship is a measure of the effect of the stress on the life.
    As the absolute value of "n" increases, the greater the effect of the stress.
    Negative values of "n" indicate an increasing life with increasing stress.
    An absolute value of "n" approaching zero indicates small effect of the stress on the life,
    with no effect (constant life with stress) when "n" = 0

    Inverse Power Law Model is used for non-thermal accelerated stresses
    :param s_acc: Accelerated Test S
    :param s_use: Normal Operating/Field Stress S
    :param n: Model Exponent
    :return:
    """
    return (s_use / s_acc) ** n

def peck_acceleration_factor(ea, n_exp,  t_use, t_acc, rh_use, rh_acc):
    """
    Temperature and Humidity Accelerated Life Model

    :param ea: Activation Energy (Ea)
    :param n_exp: Relative Humidity Exponential Factor
    :param t_use: Field Temperature (°C)
    :param t_acc: Test Temperature (°C)
    :param rh_use: Field Relative Humidity (%)
    :param rh_acc: Test Relative Humidity (%)
    :return: Acceleration Factor
    """
    return (rh_use/rh_acc)**(-n_exp) * math.exp((-ea/8.617e-5) * ((1 / (t_acc + 273.15)) - (1 / (t_use + 273.15))))


if __name__ == '__main__':
    print(f'Coffin Manson Acceleration Factor (AF): {coffin_manson_acceleration_factor(dt_acc=140, dt_use=100, cm_exp=2):.2f}')

    print(f'Arrhenius Acceleration Factor (AF): {arrhenius_acceleration_factor(t_acc=110, t_use=80, ea=.5):.2f}')
    print(f'Arrhenius Hour Conversion: {arrhenius_hours_conversion({100:1, 110:1, 120:1, 130:1, 140:1 }, 150, .5):.2f} hours')
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
    print(f'Test Cycles (Coffin Manson Model): {coffin_manson_cycles_conversion(test_dict, 60, 2)} cycles')
    print(f'Test Hours (Arrhenius Model): {arrhenius_hours_conversion(test_dict, 60, .5):.2f} hours')