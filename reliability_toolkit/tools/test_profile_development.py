
from dataclasses import dataclass
import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
# import mpl_axes_aligner
#
# from source.tools.utils import seconds_to_minutes, seconds_to_hours, minutes_to_hours, minutes_to_seconds, \
#     hours_to_seconds


@dataclass
class ChamberControls:
    '''
    temperature_ramp_up_rate: °C/min
    temperature_ramp_down_rate: °C/min
    time_interval_s: Sample Rate in seconds
    '''
    temperature_ramp_up_rate: float
    temperature_ramp_down_rate: float
    time_interval_s: int = 1

    def __post_init__(self):
        self.ramp_up_rate = self.temperature_ramp_up_rate / 60 * self.time_interval_s
        self.ramp_down_rate = self.temperature_ramp_down_rate / 60 * self.time_interval_s

@dataclass
class TemperatureProfile:
    starting_temperature: float
    temperature_set_points: dict[float, int]

@dataclass
class CurrentPulseProfile:
    current_profile: list[float]
    time_interval_s: int = 1

    def __post_init__(self):
        self.max_current = max(self.current_profile)
        self.min_current = min(self.current_profile)
        self.mean_current = np.mean(self.current_profile)

@dataclass
class CurrentCycleProfile:
    initial_delay: int
    repeat_time: int
    current_pulse_sequence: list[tuple[CurrentPulseProfile, int]]

    def __post_init__(self):
        self.profile = [0] * self.initial_delay
        for pulse in self.current_pulse_sequence:
            self.profile += pulse[0].current_profile
            self.profile += [0] * pulse[1]

        remaining_time = self.repeat_time - len(self.profile)
        self.profile += [0]*remaining_time


def chamber_temperature_cycling_profile(
        chamber_controls: ChamberControls,
        temperature_profile: TemperatureProfile,
        number_of_cycles: int = 1
):
    # Initialize Dataframe
    # Set Starting Temperature
    temperature = [temperature_profile.starting_temperature]
    status = ['Start']

    # Ramp Temperature to initial set point
    initial_set_point = list(temperature_profile.temperature_set_points.keys())[0]
    if initial_set_point > temperature[0]: # Ramp Up
        time_delta = (initial_set_point - temperature[0]) / chamber_controls.ramp_up_rate
        temperature_ramp = np.linspace(temperature[0], initial_set_point, int(time_delta))
        temperature += list(temperature_ramp)
        status += [f'Ramp Up @ {chamber_controls.temperature_ramp_up_rate}°C/min']*int(time_delta)

    elif initial_set_point < temperature[0]: # Ramp Down
         time_delta = (temperature[0] - initial_set_point) / chamber_controls.ramp_down_rate
         temperature_ramp = np.linspace(temperature[0], initial_set_point, int(time_delta))
         temperature += list(temperature_ramp)
         status += [f'Ramp Down @ {chamber_controls.temperature_ramp_down_rate}°C/min']*int(time_delta)

    # Start First Cycle / Loop through number of cycles
    for i in range(number_of_cycles):
        for set_point, time in temperature_profile.temperature_set_points.items():
            print(f'Cycle {i}, Set Point: {set_point}, Time: {time}, Temperature: {temperature[-1]}')
            if set_point > temperature[-1]: # Ramp Up
                time_delta = (set_point - temperature[-1]) / chamber_controls.ramp_up_rate
                temperature_ramp = np.linspace(temperature[-1], set_point, int(time_delta))
                temperature += list(temperature_ramp)
                status += [f'Ramp Up @ {chamber_controls.temperature_ramp_up_rate}°C/min']*int(time_delta)

            elif set_point < temperature[-1]: # Ramp Down
                time_delta = (temperature[-1] - set_point) / chamber_controls.ramp_down_rate
                temperature_ramp = np.linspace(temperature[-1], set_point, int(time_delta))
                temperature += list(temperature_ramp)
                status += [f'Ramp Down @ {chamber_controls.temperature_ramp_down_rate}°C/min']*int(time_delta)


            # Hold at set point for time
            temperature += [set_point]*time
            status += [f'Hold @ {set_point}°C']*time

    # Ramp Temperature to initial set point
    if temperature_profile.starting_temperature > temperature[-1]: # Ramp Up
        time_delta = (temperature_profile.starting_temperature - temperature[-1]) / chamber_controls.ramp_up_rate
        temperature_ramp = np.linspace(temperature[-1], temperature_profile.starting_temperature, int(time_delta))
        temperature += list(temperature_ramp)
        status += [f'Ramp Up @ {chamber_controls.temperature_ramp_up_rate}°C/min']*int(time_delta)

    elif temperature_profile.starting_temperature < temperature[-1]: # Ramp Down
        time_delta = (temperature[-1] - temperature_profile.starting_temperature) / chamber_controls.ramp_down_rate
        temperature_ramp = np.linspace(temperature[-1], temperature_profile.starting_temperature, int(time_delta))
        temperature += list(temperature_ramp)
        status += [f'Ramp Down @ {chamber_controls.temperature_ramp_down_rate}°C/min']*int(time_delta)

    temperature = pd.DataFrame({'Temperature': temperature, 'Status': status})
    temperature.Status = temperature.Status.astype('category')
    temperature['Time'] = temperature.index * chamber_controls.time_interval_s
    temperature['Chamber_Step_ID'] = temperature.Status.ne(temperature.Status.shift()).cumsum()
    print(temperature)

    status = temperature.groupby(['Status','Chamber_Step_ID'], observed=True)[['Time','Temperature']].mean().sort_values('Chamber_Step_ID').reset_index()
    print(status)

    return temperature, status


if __name__ == "__main__":
    pass


