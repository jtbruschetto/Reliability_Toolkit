# Acceleration Models

## Overview

This script provides functions for calculating acceleration factors and converting cycles or hours based on different acceleration models:

- Coffin-Manson Model (for Temperature Cycling Tests)
- Arrhenius Model (for Temperature/Time Tests)
- Inverse Power Law Model (for Non-Thermal Stresses)
- Peck Model (for Temperature and Humidity)

## Functions

### Coffin-Manson Model

- `coffin_manson_acceleration_factor(dt_acc, dt_use, cm_exp)`: Calculates the acceleration factor for a Coffin-Manson equation.
- `coffin_manson_cycles_conversion(cycles, dt_eval, cm_exp)`: Converts a stress distribution into an equivalent number of cycles at a given stress set point.

### Arrhenius Model

- `arrhenius_acceleration_factor(ea, t_acc, t_use)`: Calculates the acceleration factor for an Arrhenius equation.
- `arrhenius_hours_conversion(hours, t_eval, ea)`: Converts a field temperature distribution into test hours.

### Inverse Power Law Model

- `inverse_power_law_acceleration_factor(s_acc, s_use, n)`: Calculates the acceleration factor for an inverse power law equation.

### Peck Model

- `peck_acceleration_factor(ea, n_exp, t_use, t_acc, rh_use, rh_acc)`: Calculates the acceleration factor for a temperature and humidity accelerated life model.

## Usage

The script includes example usage in the `if __name__ == '__main__':` block. This demonstrates how to use each function and provides sample outputs.

Feel free to modify and use this script for your acceleration modeling needs.