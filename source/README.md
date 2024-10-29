# Reliability_Toolkit

## Table of Contents
1. [Summary](#summary)
2. [Functions](#functions)
   1. [Accelerated Reliability Models](#Accelerated_Reliability)
      1. [Coffin-Manson](#Coffin-Manson_Accelerated_Reliability_Model)
   2. [Classical Test Design](#Classical Test Design)
   3. [Reliability Dataclasses](#Reliability Dataclasses)
   4. [Reliability Demonstration](#Reliability Demonstration)
   5. [Reliability Functions](#Reliability Functions)
   6. [Reliability Target Conversion](#Reliability Target Conversion)
   7. [System Reliability](#System Reliability)
   8. [Time Series Data Analysis](#Time Series Analysis)
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
 $$AF = (dt_{acc} / dt_{use}) ^{cm_{exp}}$$

**Input Parameters:**

| Parameter    | Description |
|--------------| ----------- |
| $$dt_{acc}$$ | Temperature delta exercised in the accelerated test |
| $$dt_{use}$$ | Temperature delta seen in the field |
| $$cm_{exp}$$ | Coffin Manson Exponent |

**Output:**

| Variable | Description |
|----------| ----------- |
| $$AF$$   | Acceleration Factor |

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
$$ Equivalent Cycles = \sum_{i=1}^{n} \frac{Count_{i}}{(dt_{eval} / dt_{i}) ^{cm_{exp}}} $$

**Input Parameters:**

| Parameter    | Description                                     |
|--------------|-------------------------------------------------|
| $$cycles$$   | Dictionary of temperature delta's and counts { dt as Key: Count as Value } |
| $$dt_{eval}$$ | Stress Set Point for evaluation                 |
| $$cm_{exp}$$ | Coffin Manson Exponent                          |


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




#### Arrhenius Accelerated Reliability Model <a name="Arrhenius Accelerated Reliability Model"></a>

#### Inverse Power Law Accelerated Reliability Model <a name="Inverse Power Law Accelerated Reliability Model"></a>

#### Peck Accelerated Reliability Model <a name="Peck Accelerated Reliability Model"></a>

### Classical Test Design <a name="Classical Test Design"></a>

### Reliability Dataclasses <a name="Reliability Dataclasses"></a>

### Reliability Demonstration <a name="Reliability Demonstration"></a>

### Reliability Functions <a name="Reliability Functions"></a>

### Reliability Target Conversion <a name="Reliability Target Conversion"></a>

### System Reliability <a name="System Reliability"></a>

### Time Series Data Analysis <a name="Time Series Analysis"></a>

### Utilities <a name="Utilities"></a>

## Reliability Toolkit Apps <a name="apps"></a>

### Classical Test Design App <a name="Classical Test Design App"></a>

### Reliability/MRpTV Conversion App <a name="Reliability/MRpTV App"></a>

## Reliability Toolkit References <a name="references"></a>