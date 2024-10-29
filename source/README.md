# Reliability_Toolkit

## Table of Contents
1. [Summary](#summary)
2. [Functions](#functions)
   1. [Accelerated Reliability Models](#Accelerated_Reliability)
      1. [Coffin-Manson](#Coffin-Manson_Accelerated_Reliability_Model)
      2. [Arrhenius](#Arrhenius_Accelerated_Reliability_Model)
      3. [Inverse Power Law](#Inverse_Power_Law_Accelerated_Reliability_Model)
      4. [Peck](#Peck_Accelerated_Reliability_Model)
   2. [Classical Test Design](#Classical_Test_Design)
      1. [Calculate Reliability](#Calculate_Reliability)
      2. [Calculate Confidence](#Calculate_Confidence)
      3. [Calculate Life Ratio](#Calculate_Life_Ratio)
      4. [Calculate Sample Size](#Calculate_Sample_Size)
      5. [Calculate Allowable Number of Failures](#Calculate_Allowable_Number_of_Failures)
   3. [Reliability Dataclasses](#Reliability_Dataclasses)
   4. [Reliability Demonstration](#Reliability_Demonstration)
   5. [Reliability Functions](#Reliability_Functions)
   6. [Reliability Target Conversion](#Reliability_Target_Conversion)
   7. [System Reliability](#System_Reliability)
   8. [Time Series Data Analysis](#Time_Series_Analysis)
   9. [Utilities](#Utilities)
3. [Apps](#apps)



## Reliability Toolkit Summary <a name="summary"></a>

Reliability Toolkit is a collection of Python scripts and tools that can be used to calculate Reliability.

## Reliability Toolkit Functions <a name="functions"></a>

### Accelerated Reliability Models <a name="Accelerated_Reliability"></a>

#### Coffin Manson Accelerated Reliability Model <a name="Coffin-Manson_Accelerated_Reliability_Model"></a>
##### Coffin Manson Acceleration Factor (AF) Function
The Coffin Manson Accelerated Reliability Model is used for Temperature Cycling Tests

**Function Name:** `coffin_manson_acceleration_factor`

**Function Equation:**
$$AF = (\delta T_{acc} / \delta T_{use}) ^{cm_{exp}}$$

**Input Parameters:**

| Parameter    | Description |
|--------------| ----------- |
|$$\delta T_{acc}$$ | Temperature delta exercised in the accelerated test |
|$$\delta T_{use}$$ | Temperature delta seen in the field |
|$$cm_{exp}$$ | Coffin Manson Exponent |

**Output:**

| Variable | Description |
|----------| ----------- |
|$$AF$$   | Acceleration Factor |

**Example:**

```python
from source.tools.acceleration_models import *
coffin_manson_acceleration_factor(dt_acc=140, dt_use=100, cm_exp=2)
```

#### Coffin Manson Equivalent Cycle Conversion
The Coffin Manson Equivalent Cycle Conversion is used for converting a stress distribution into equivalent number of cycles at a given stress set point

**Function Name:** `coffin_manson_cycles_conversion`

**Function Description:**
Convert a dictionary of temperature delta's and counts to equivalent number of cycles at a given stress set point

**Function Equation:**
$$ Equivalent Cycles = \sum_{i=1}^{n} \frac{Count_{i}}{(\delta T_{eval} / \delta T_{i}) ^{cm_{exp}}}$$

**Input Parameters:**

| Parameter    | Description                                                                       |
|--------------|-----------------------------------------------------------------------------------|
|$$cycles$$   | Dictionary of temperature delta's and counts { $\delta$T as Key: Count as Value } |
|$$\delta T_{eval}$$ | Stress Set Point for evaluation                                                   |
|$$cm_{exp}$$ | Coffin Manson Exponent                                                            |


**Example:**
```python
from source.tools.acceleration_models import *
cycles = {
   # Temperature Delta as Key: Count of occurrences as Value
   100: 5,
   120: 3,
   140: 1
}
coffin_manson_cycles_conversion(cycles, dt_eval=100, cm_exp=2)
```

#### Arrhenius Accelerated Reliability Model <a name="Arrhenius_Accelerated_Reliability_Model"></a>
##### Arrhenius Acceleration Factor (AF) Function

**Function Name:** `arrhenius_acceleration_factor`

**Function Equation:**
$$AF = \exp(\frac{-EA}{k}(\frac{1}{T_{use}+ 273.15} - \frac{1}{T_{acc}+ 273.15}))$$

**Input Parameters:**

| Parameter    | Description |
|--------------| ----------- |
|$$EA$$       | Activation Energy (ev) |
|$$T_{acc}$$  | Test Temperature (°C) |
|$$T_{use}$$  | Field Temperature (°C) |

**Output:**

| Variable | Description |
|----------| ----------- |
|$$AF$$   | Acceleration Factor |

**Example:**

```python
from source.tools.acceleration_models import *
arrhenius_acceleration_factor(ea=40, t_acc=0, t_use=57)
```

##### Arrhenius Hour Conversion

**Function Name:** `arrhenius_hour_conversion` 

**Function Description:**
Convert a dictionary of temperature's and time at temperature in hours to equivalent number of hours at a given stress set point

**Function Equation:**
$$H_{equivalent}=\sum_{i=1}^{n}\frac{Hours_{i}}{\exp(\frac{-EA}{k}(\frac{1}{T_{i}+273.15}-\frac{1}{T_{eval}+273.15}))}$$

**Input Parameters:**

| Parameter    | Description                                     |
|--------------|-------------------------------------------------|
|$$hours$$    | Dictionary of temperature's and time at temperature in hours { Temperature as Key: Time at Temperature in Hours as Value } |
|$$T_{eval}$$ | Stress Set Point for evaluation                 |
|$$EA$$       | Activation Energy (ev)                          |

**Output:**

| Variable | Description |
|----------| ----------- |
|$$H_{equivalent}$$   | Equivalent Number of Hours |

**Example:**

```python
from source.tools.acceleration_models import *
hours = {
   # Temperature as Key: Time at Temperature in Hours as Value
   0: 5,
   10: 3,
   20: 1
}
arrhenius_hours_conversion(hours, t_eval=10, ea=.5)
```

#### Inverse Power Law Accelerated Reliability Model <a name="Inverse_Power_Law_Accelerated_Reliability_Model"></a>

**Function Name:** `inverse_power_law_acceleration_factor`

**Function Equation:**
$$AF = \frac{S_{use}}{S_{acc}}^{n}$$

**Input Parameters:**

| Parameter    | Description |
|--------------| ----------- |
|$$S_{acc}$$  | Accelerated Test S |
|$$S_{use}$$  | Normal Operating/Field Stress S |
|$$n$$        | Model Exponent |

**Output:**

| Variable | Description |
|----------| ----------- |
|$$AF$$   | Acceleration Factor |

**Example:**

```python
from source.tools.acceleration_models import *
inverse_power_law_acceleration_factor(s_acc=100, s_use=50, n=2)
```

#### Peck Accelerated Reliability Model <a name="Peck_Accelerated_Reliability_Model"></a>

**Function Name:** `peck_acceleration_factor`

**Function Equation:**
$$AF = \frac{RH_{use}}{RH_{acc}} * \exp(\frac{-EA}{k}(\frac{1}{T_{use}+ 273.15} - \frac{1}{T_{acc}+ 273.15}))$$

**Input Parameters:**

| Parameter    | Description |
|--------------| ----------- |
|$$EA$$       | Activation Energy (ev) |
|$$T_{acc}$$  | Test Temperature (°C) |
|$$T_{use}$$  | Field Temperature (°C) |
|$$RH_{use}$$ | Field Relative Humidity (%) |
|$$RH_{acc}$$ | Test Relative Humidity (%) |

**Output:**

| Variable | Description |
|----------| ----------- |
|$$AF$$   | Acceleration Factor |

**Example:**

```python
from source.tools.acceleration_models import *
peck_acceleration_factor(ea=40, t_acc=85, t_use=40, rh_use=.50, rh_acc=.95)
```

### Classical Test Design <a name="Classical_Test_Design"></a>

#### Classical Test Design Class <a name="Classical_Test_Design_Class"></a>

#### Calculate Reliability <a name="Calculate_Reliability"></a>
**Function Name:** `calculate_reliability`

**Equation:**
$$R = \exp(\frac{-\chi^2_{CI, 2f+2}}{2 * n * L_v^{\beta}})$$

**Input Parameters:**

| Parameter | Description         |
|-----------|---------------------|
| $CI$      | Confidence Interval |
| $f$       | Number of Failures  |
| $n$       | Sample Size         |
| $L_v$     | Life Ratio          |
| $\beta$   | Weibull Shape       |

**Output:**

| Variable | Description |
|----------|-------------|
| $R$      | Reliability |

**Example:**

```python
from source.tools.classical_test_design import *
calculate_reliability(ci=0.95, f=2, n=100, lv=1, beta=2)
```


#### Calculate Confidence Interval <a name="Calculate_Confidence_Interval"></a>
**Function Name:** `calculate_confidence`

**Equation:**
$$CI = \chi^2_{-2 * n * L_v^{\beta} *\log(R), 2f+2}$$

**Input Parameters:**

| Parameter | Description         |
|-----------|---------------------|
| $f$       | Number of Failures  |
| $n$       | Sample Size         |
| $L_v$     | Life Ratio          |
| $\beta$   | Weibull Shape       |

**Output:**

| Variable | Description |
|----------|-------------|
| $CI$     | Confidence Interval |

**Example:**

```python
from source.tools.classical_test_design import *
calculate_confidence(f=2, n=100, lv=1, beta=2)
```

#### Calculate Sample Size <a name="Calculate_Sample_Size"></a>
**Function Name:** `calculate_sample_size`

**Equation:**
$$n = \frac{\chi^2_{CI, 2f+2}}{2 * L_v^{\beta} * \log(R)}$$

**Input Parameters:**

| Parameter | Description         |
|-----------|---------------------|
| $CI$      | Confidence Interval |
| $f$       | Number of Failures  |
| $L_v$     | Life Ratio          |
| $\beta$   | Weibull Shape       |

**Output:**

| Variable | Description |
|----------|-------------|
| $n$      | Sample Size |


**Example:**

```python
from source.tools.classical_test_design import *
calculate_sample_size(confidence=0.95, failures=2, life_ratio=1, reliability=0.9, beta=2)
```

#### Calculate Life Ratio <a name="Calculate_Life_Ratio"></a>

**Function Name:** `calculate_life_ratio`

**Equation:**
$$L_v = (\frac{\chi^2_{CI, 2f+2}}{2 * n * \log(R)})^{\frac{1}{\beta}}$$

**Input Parameters:**

| Parameter | Description         |
|-----------|---------------------|
| $CI$      | Confidence Interval |
| $f$       | Number of Failures  |
| $n$       | Sample Size         |
| $\beta$   | Weibull Shape       |

**Output:**

| Variable | Description |
|----------|-------------|
| $L_v$    | Life Ratio  |

**Example:**

```python
from source.tools.classical_test_design import *
calculate_life_ratio(confidence=0.95, failures=2, sample_size=100, reliability=0.9, beta=2)
```

#### Calculate Allowable Number of Failures <a name="Calculate_Allowable_Number_of_Failures"></a>

***TODO***

### Reliability Dataclasses <a name="Reliability_Dataclasses"></a>
***TODO***

### Reliability Demonstration <a name="Reliability_Demonstration"></a>
***TODO***

### Reliability Functions <a name="Reliability_Functions"></a>
***TODO***

### Reliability Target Conversion <a name="Reliability_Target_Conversion"></a>
***TODO***

### System Reliability <a name="System_Reliability"></a>
***TODO***

### Time Series Data Analysis <a name="Time_Series_Analysis"></a>
***TODO***

### Utilities <a name="Utilities"></a>
***TODO***

## Reliability Toolkit Apps <a name="apps"></a>
***TODO***

### Classical Test Design App <a name="Classical Test Design App"></a>
***TODO***

### Reliability/MRpTV Conversion App <a name="Reliability/MRpTV App"></a>
***TODO***

## Reliability Toolkit References <a name="references"></a>
***TODO***