from cProfile import label
from dataclasses import dataclass
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import mpl_axes_aligner

from source.tools.utils import seconds_to_minutes, seconds_to_hours, minutes_to_hours, minutes_to_seconds, \
    hours_to_seconds


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

    temperature_profile, status = chamber_temperature_cycling_profile(
        ChamberControls(
            temperature_ramp_up_rate=3, # °C/min
            temperature_ramp_down_rate=1.5, # °C/min
            time_interval_s=1
        ),
        TemperatureProfile(
            starting_temperature=20, # °C
            temperature_set_points={-40: 25*60,80: 5*60 } # {°C Set Point: Dwell Time (s)}
        ),
    )
    print(f'Thermal Cycle Duration {temperature_profile.Time.max()}')
    print(f'Time To -20 {temperature_profile[(temperature_profile.Temperature<-24) & (temperature_profile.Temperature >-25)]}')
    constant_low_current_cycle_long = CurrentPulseProfile(current_profile= [400] * 600) # [Current Value] * Duration(s)
    constant_low_current_cycle_short = CurrentPulseProfile(current_profile=[400] * 30)  # [Current Value] * Duration(s)
    constant_high_current_cycle = CurrentPulseProfile(current_profile= [800] * 150)  # [Current Value] * Duration(s)
    wot_cycle = CurrentPulseProfile(current_profile= [
        459,
        1101.2,
        1869.2,
        1887.9,
        1846.8,
        1796.4,
        1754.1,
        1726.9,
        1708.1,
        1693.1,
        1679.5,
        1663.7,
        1644.7,
        404.6,
        478,
        480.3,
        483.1,
        484.9,
        486.5,
        488.2,
        488.7,
        476.5,
        476.4,
        475.4,
        471.4,
        462.9,
        450.6,
        433.8,
        398.9,
        363.2,
        326.8,
        287.6,
        248.9,
        209.4,
        167.2,
        124.9,
        83.2,
        39.6,
    ]) # Custom Wide Open Throttle Profile
    repeat_cycle = CurrentPulseProfile(current_profile=[850]*10)  # [Current Value] * Duration(s)
    current_profile = CurrentCycleProfile(
        initial_delay = 4200,
        repeat_time= 9001,
        current_pulse_sequence=[
            (constant_low_current_cycle_long, 300),
            (constant_high_current_cycle, 150),
            (constant_high_current_cycle, 150),
            (wot_cycle, 30),
            (constant_low_current_cycle_short, 30),
            (wot_cycle, 30),
            (constant_low_current_cycle_short, 30),
            (wot_cycle, 30),
            (constant_low_current_cycle_short, 120),
            (constant_high_current_cycle, 150),
            (repeat_cycle, 30),
            (repeat_cycle, 30),
            (repeat_cycle, 30),
            (repeat_cycle, 30),
            (repeat_cycle, 30),
        ]
    )
    temperature_profile['Current_Profile'] = current_profile.profile


    fig, ax = plt.subplots()

    axc = ax.twinx()
    temperature_profile.plot(x='Time', y=['Current_Profile'], ax=axc, c='orange', xlabel='time (s)', ylabel='Current (°A)', title='PTC Profile')
    temperature_profile.plot(x='Time', y=['Temperature'], ax=ax, c='r', xlabel='time (s)', ylabel='Temperature (°C)', title='PTC Profile')
    minax = ax.secondary_xaxis(-.15, functions=(seconds_to_minutes, minutes_to_seconds))
    minax.set_xlabel('time (min)')
    hrax = ax.secondary_xaxis(1, functions=(seconds_to_hours, hours_to_seconds))
    hrax.set_xlabel('time (hour)')

    for i, row in status.iterrows():
        if 'Ramp' in row['Status']:
            ax.annotate(row['Status'].replace('@','\n'), (row['Time'], row['Temperature']),
                     horizontalalignment='left', verticalalignment='center', fontsize=8)
        else:
            ax.annotate(row['Status'], (row['Time'], row['Temperature']),
                     horizontalalignment='center', verticalalignment='bottom', fontsize=8)
    mpl_axes_aligner.align.yaxes(ax, 0, axc, 0, 0.25)
    ax.legend(loc=2)
    axc.legend(loc=1)
    plt.tight_layout()
    plt.show()

    temperature_profile['28s_rolling_rms'] = temperature_profile['Current_Profile'].rolling(28).apply(lambda x: np.sqrt(np.mean(x**2)))
    temperature_profile['70s_rolling_rms'] = temperature_profile['Current_Profile'].rolling(70).apply(lambda x: np.sqrt(np.mean(x ** 2)))
    temperature_profile['290s_rolling_rms'] = temperature_profile['Current_Profile'].rolling(290).apply(lambda x: np.sqrt(np.mean(x**2)))
    temperature_profile['880s_rolling_rms'] = temperature_profile['Current_Profile'].rolling(880).apply(lambda x: np.sqrt(np.mean(x**2)))


    temperature_profile.plot(x='Time', y=['28s_rolling_rms','70s_rolling_rms','290s_rolling_rms', '880s_rolling_rms'], xlabel='time (s)', ylabel='RMS Current (A)', title='PTC RMS Current')
    plt.axhline(y=1500, label='28s RMS Limit', c='b')
    plt.axhline(y=1000, label='70s RMS Limit', c='orange')
    plt.axhline(y=600, label='290s RMS Limit', c='g')
    plt.axhline(y=500, label='880s RMS Limit', c='r')
    plt.legend()
    plt.show()
    temperature_profile.to_csv('ls_contactor_ptc_profile_v1.csv')


