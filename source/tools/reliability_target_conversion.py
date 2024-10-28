import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal

from dataclasses import dataclass

from numpy import linspace
from scipy.optimize import curve_fit

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

''' 
MRR Calculations


- Reliability(t)
    -> MRpTV 
    -> Reliability
    -> UnReliability
    -> Reliability(t1)
- MRpTV(t)
    -> MRpTV 
    -> Reliability
    -> UnReliability
    -> Reliability(t1)

'''

@dataclass
class ReliabilityAtYear:
    reliability: float
    beta: float
    gamma: int = 0
    reliability_year: int = 10
    mode: Literal['at_year', 'upto_year'] = 'upto_year'

    def __post_init__(self):
        self.eta = calculate_eta_from_reliability(r_ts=self.reliability, beta=self.beta, gamma=self.gamma, ts=self.reliability_year)
        self.reliability_table = calculate_reliability_from_mrr(beta=self.beta, eta=self.eta, gamma=self.gamma, mode=self.mode)
        self.reliability_table = self.reliability_table.join(calculate_unreliability_by_year(beta=self.beta, eta=self.eta, gamma=self.gamma))
        self.reliability_table = self.reliability_table.join(calculate_failure_rate_by_year(beta=self.beta, eta=self.eta, gamma=self.gamma))

    def get_reliability_at_year(self, year):
        return self.reliability_table.at[year, 'Reliability_at_year']

    def get_reliability_by_year(self):
        return self.reliability_table['Reliability_at_year']

    def get_unreliability_at_year(self, year):
        return self.reliability_table.at[year, 'UnReliability']

    def get_unreliability_by_year(self):
        return self.reliability_table['UnReliability']

    def get_failure_rate_at_year(self, year):
        return self.reliability_table.at[year, 'Failure_Rate']

    def get_failure_rate_by_year(self):
        return self.reliability_table['Failure_Rate']

    def get_mrptv_at_year(self, year):
        return self.reliability_table.at[year, f'MRpTV_{self.mode}']

    def get_mrptv_by_year(self):
        return self.reliability_table[f'MRpTV_{self.mode}']

    def get_mrr_at_year(self, year):
        return self.reliability_table.at[year, f'MRR_{self.mode}']

    def get_mrr_by_year(self):
        return self.reliability_table[f'MRR_{self.mode}']


@dataclass
class MRpTVAtYear:
    mrptv: float
    mrptv_year: int
    beta: float
    gamma: int = 0
    mode: Literal['at_year', 'upto_year'] = 'upto_year'

    def __post_init__(self):
        self.mrr = self.mrptv / 1_000
        self.eta = eta_mrr_optimization(beta=self.beta, mrr_year=self.mrptv_year, mrr_desired=self.mrr, gamma=0, mode=self.mode, show_plot=True)
        self.reliability_table = calculate_reliability_from_mrr(beta=self.beta, eta=self.eta, gamma=self.gamma, mode=self.mode)
        self.reliability_table = self.reliability_table.join(calculate_unreliability_by_year(beta=self.beta, eta=self.eta, gamma=self.gamma))
        self.reliability_table = self.reliability_table.join(calculate_failure_rate_by_year(beta=self.beta, eta=self.eta, gamma=self.gamma))

    def get_reliability_at_year(self, year):
        return self.reliability_table.at[year, 'Reliability_at_year']

    def get_reliability_by_year(self):
        return self.reliability_table['Reliability_at_year']

    def get_unreliability_at_year(self, year):
        return self.reliability_table.at[year, 'UnReliability']

    def get_unreliability_by_year(self):
        return self.reliability_table['UnReliability']

    def get_failure_rate_at_year(self, year):
        return self.reliability_table.at[year, 'Failure_Rate']

    def get_failure_rate_by_year(self):
        return self.reliability_table['Failure_Rate']

    def get_mrptv_at_year(self, year):
        return self.reliability_table.at[year, f'MRpTV_{self.mode}']

    def get_mrptv_by_year(self):
        return self.reliability_table[f'MRpTV_{self.mode}']

    def get_mrr_at_year(self, year):
        return self.reliability_table.at[year, f'MRR_{self.mode}']

    def get_mrr_by_year(self):
        return self.reliability_table[f'MRR_{self.mode}']



def calculate_eta_from_reliability(r_ts: float, beta: float, gamma: float, ts: int):
    """

    :param r_ts: Reliability target
    :param beta: Component Weibull Shape parameter.
    :param gamma: Weibull Location Parameter
    :param ts: Time in Service (years)
    :return:
    """
    return (ts * 12 - gamma) / ((-1 * np.log(r_ts)) ** (1 / beta))


def calculate_reliability_from_mrr(beta: float, eta: float, gamma:float=0, quantity:int=1000000, years:int=15,
                                   mode: Literal['at_year', 'upto_year'] = 'upto_year', **_kwargs):
    """

    :param beta: Weibull Shape Parameter
    :param eta: Weibull Scale Parameter
    :param gamma: Weibull Location Parameter
    :param years: Compute results up to year
    :param mode: 'at_year' or 'upto_year'
    :param quantity: Build Quantity (Placeholder)

    :return: Dictionary with MRR_at_year, MRR_upto_year, and DataFrame Summary Table
    """
    build_month = range(1, (years * 12) + 1, 1)

    df_support = (pd.DataFrame(index=build_month, columns=["delivered_quantity", "adjusted_quantity"])
                  .astype(float).fillna(0.0))
    df_support.at[1, "delivered_quantity"] = quantity
    df_support.at[1, "adjusted_quantity"] = quantity

    results = {}

    for bm in build_month:
        results[bm] = []
        sum_reduction = 0
        for om in build_month:
            if bm > om:
                r = 0
            else:
                r = mrr_calculation_function(beta, eta, gamma, om, df_support.at[bm, "adjusted_quantity"],
                                             sum_reduction)
            results[bm].append(r)
            sum_reduction = sum(results[bm])

        # df_table[bm] = results[bm]
        if bm <= max(build_month) - 1:
            df_support.at[bm + 1, "adjusted_quantity"] = sum(results[bm]) + df_support.at[bm + 1, "delivered_quantity"]

    df_table = pd.DataFrame(results)
    df_table.index = np.arange(1, len(df_table) + 1)

    df_support['accumulated_quantity'] = df_support.delivered_quantity.cumsum()
    df_support['r_calendar_month'] = df_table.sum(axis=1)
    df_support['MRR_at_year'] = df_support['r_calendar_month'] / df_support['accumulated_quantity']
    df_support['accumulated_r'] = df_support['r_calendar_month'].cumsum()
    df_support['accumulatedx2_quantity'] = df_support['accumulated_quantity'].cumsum()
    df_support['accumulated_MRR'] = df_support['accumulated_r'] / df_support['accumulatedx2_quantity']
    df_support['year'] = (df_support.index - 1) // 12 + 1
    df_summary = df_support.groupby(by='year').mean()
    df_summary = df_summary.drop(
        columns=['delivered_quantity', "adjusted_quantity", 'accumulated_quantity', 'r_calendar_month', 'accumulated_r',
                 'accumulatedx2_quantity', 'accumulated_MRR'])
    if mode == 'upto_year':
        df_summary['MRR_upto_year'] = df_summary['MRR_at_year'].rolling(len(df_summary.index), 0).mean()
        df_summary['MRpTV_upto_year'] = df_summary['MRR_upto_year'] * 1000
        df_summary = df_summary.drop(columns=['MRR_at_year'])
    else:
        df_summary['MRpTV_at_year'] = df_summary['MRR_at_year'] * 1000

    df_summary['Reliability_at_year'] = np.exp(0 - ((12 * df_summary.index - gamma) / eta) ** beta)
    return df_summary


def mrr_calculation_function(beta: float, eta: float, gamma: float, month: int, quantity: int, sum_reduction,
                             **_kwargs):
    """

    :param beta: Weibull Shape Parameter
    :param eta: Weibull Scale Parameter
    :param gamma: Weibull Location Parameter
    :param month: Month since build date
    :param quantity: Build Quantity (Placeholder)
    :param sum_reduction: Sum of Repair Events up to Current Month
    :return: Vehicle Repairs of a given month
    """

    if month - 1 - gamma < 0:
        return 0
    else:
        return ((quantity - sum_reduction)
            * (np.exp(0 - ((month - 1 - gamma) / eta) ** beta) - np.exp(0 - ((month - gamma) / eta) ** beta))
            / np.exp(0 - ((month - 1 - gamma) / eta) ** beta))


def calculate_mrr_from_reliability(beta: float, eta: float, eta_lb=-0.1, eta_ub=-0.1, gamma=0, years=15,
                                   mode: Literal['at_year', 'upto_year'] = 'upto_year', **_kwargs):
    """

    :param beta:  Component Weibull Shape parameter.
    :param eta: Weibull Scale Parameter
    :param eta_lb: OPTIONAL - Weibull Scale Parameter (Lower Bound Confidence)
    :param eta_ub: OPTIONAL - Weibull Scale Parameter (Upper Bound Confidence)
    :param gamma: Weibull Location Parameter
    :param years: Compute results up to year
    :param mode: 'at_year' or 'upto_year'
    :return:
    """
    year_range = range(1, years + 1, 1)
    df = pd.DataFrame(index=year_range)
    eta_list = [eta, eta_lb, eta_ub]
    eta_list = [x for x in eta_list if x >= 1]
    for eta_run in eta_list:
        inputs = {
            'beta': beta,
            'eta': eta_run,
            'gamma': gamma,
            'years': years,
            'mode': mode
        }

        if eta_run == eta:
            col_designator = ""
        elif eta_run == eta_lb:
            col_designator = "_LowerBound"
        elif eta_run == eta_ub:
            col_designator = "_UpperBound"
        else:
            col_designator = ""
        ur_df = calculate_unreliability_by_year(**inputs)
        df = df.join(ur_df, rsuffix=col_designator)

        mrr_output = calculate_reliability_from_mrr(year=years, **inputs)
        mrr_df = mrr_output.drop(columns=['Reliability_at_year'])
        df = df.join(mrr_df, rsuffix=col_designator)

        fr_df = calculate_failure_rate_by_year(**inputs)
        df = df.join(fr_df, rsuffix=col_designator)
    df = df.reindex(sorted(df.columns), axis=1)
    return df


def calculate_failure_rate_by_year(beta: float, eta: float, gamma: float = 0, years=15, **_kwargs):
    """

    :param beta: Component Weibull Shape parameter.
    :param eta: Weibull Scale Parameter
    :param gamma: Weibull Location Parameter
    :param years: Compute results up to year
    :return:
    """
    month_range = range(1, (years * 12) + 1, 1)
    df = pd.DataFrame(index=month_range)
    df['Failure_Rate'] = (beta / eta) * ((df.index - gamma) / eta) ** (beta - 1)
    df['year'] = (df.index - 1) // 12 + 1
    df = df.groupby(by='year').mean()
    return df


def calculate_unreliability_by_year(beta: float, eta: float, gamma: float = 0, years=15, **_kwargs):
    """
    :param beta: Component Weibull Shape parameter.
    :param eta: Weibull Scale Parameter
    :param gamma: Weibull Location Parameter
    :param years: Compute results up to year
    :return:
    """
    year_range = range(1, years + 1, 1)
    df = pd.DataFrame(index=year_range)
    df['UnReliability'] = 1 - np.exp(0 - ((df.index * 12 - gamma) / eta) ** beta)
    return df


def calculate_reliability_at_year_from_mrr(beta: float, eta: float, year=10,  gamma=0, **_kwargs):
    """

    :param mrr_desired: Steady state monthly repair rate
    :param beta:  Component Weibull Shape parameter.
    :param eta: Weibull Scale Parameter
    :param gamma: Weibull Location Parameter
    :param mrr_year:  Time interval of interest, years
    :param year:  Time interval of interest, years
    :return: Reliability
    """
    output = calculate_reliability_from_mrr(beta=beta, eta=eta, gamma=gamma)
    return output.at[year, 'Reliability_at_year']


def eta_mrr_optimization(beta, mrr_year, mrr_desired, gamma=0, quantity=1000000000,
                         mode: Literal['at_year', 'upto_year'] = 'upto_year',
                         show_plot=False, **_kwargs):
    """

    :param beta: Weibull Shape Parameter
    :param gamma: Weibull Location Parameter (preset=0)
    :param mrr_year: MRR Target Year (10 years)
    :param mrr_desired: MRR Desired Target at Specified Year
    :param mode: MRR_at_year / MRR_upto_year
    :param quantity: Build Quantity (Placeholder)
    :param show_plot: Boolean Option to Show Eta Optimization
    :return:
    """
    optimization_dict = {
        1000: 300,
        100: 150,
        50: 100,
        1: 50,
        0.1: 15,
        0.05: 5,
        0.01: 1,
        0.0025: 0.5,
        0: .1
    }
    eta = 25
    mrr_eval = 0
    past_eval = 100000
    mrr = []
    eta_range = []
    mrr_col = 'MRR_' + mode

    '''
    Notes:
        start by running initial guess - mrr_eval
        compare to mrr_desired
    '''
    while abs(mrr_eval - mrr_desired) / mrr_desired >= 0.001:
        output = calculate_reliability_from_mrr(beta=beta, eta=eta, mrr_year=mrr_year, gamma=gamma, quantity=quantity, mode=mode)
        mrr_eval = output.at[mrr_year, mrr_col]
        mrr.append(mrr_eval)
        eta_range.append(eta)
        pres_eval = abs(mrr_eval - mrr_desired) / mrr_desired
        # print(f'Eta = {eta} | Eval Loop: {pres_eval} | ({mrr_eval} - {mrr_desired}) / {mrr_desired}')
        # print(f'present_eval: {pres_eval}')


        eta_modifier = [x for x in optimization_dict.keys() if x < pres_eval][0]
        # print(f'eta_modifier: {eta_modifier}')
        eta += optimization_dict[eta_modifier]

        if past_eval < pres_eval:
            break
        past_eval = pres_eval

    output = calculate_reliability_from_mrr(beta=beta, eta=eta, gamma=gamma, quantity=quantity, mode=mode)
    mrr.append(output.at[mrr_year,mrr_col])
    eta_range.append(eta)

    df = pd.DataFrame({'eta': eta_range, mrr_col: mrr})
    # print(df)
    df_closest = df.iloc[(df[mrr_col] - mrr_desired).abs().argsort()[:1]]

    if show_plot:
        mrr_plot = df.plot(x='eta', y=f'MRR_{mode}', marker='o', logy=True,
                           title='Eta selection for desired MRR target')
        mrr_plot.axhline(y=mrr_desired, color='r')
        mrr_plot.set_ylabel('MRR (log scale)')
        mrr_plot.text(x=min(eta_range), y=mrr_desired, s=mrr_desired, horizontalalignment='left',
                      verticalalignment='bottom')
        plt.show()

    # print(f'Optimized Eta: {df_closest['eta'].values[0]}')

    return df_closest['eta'].values[0]


def eta_mrr_optimization_trial(beta, mrr_year, mrr_desired, gamma=0, quantity=1000000000,
                         mode: Literal['at_year', 'upto_year'] = 'upto_year',
                         show_plot=False, **_kwargs):
    """

    :param beta: Weibull Shape Parameter
    :param gamma: Weibull Location Parameter (preset=0)
    :param mrr_year: MRR Target Year (10 years)
    :param mrr_desired: MRR Desired Target at Specified Year
    :param mode: MRR_at_year / MRR_upto_year
    :param quantity: Build Quantity (Placeholder)
    :param show_plot: Boolean Option to Show Eta Optimization
    :return:
    """


    eta_dict = {}
    eta = 10
    # Increase eta until mrr_desired is met
    while True:
        eta_dict[eta] = calculate_reliability_from_mrr(beta=beta, eta=eta, mrr_year=mrr_year, gamma=gamma, quantity=quantity, mode=mode).at[mrr_year, 'MRR_' + mode]
        if eta_dict[float(eta)] <= mrr_desired:
            break
        else:
            eta *= 10

    # Increase Inspection Points between last two eta values
    for eta in linspace(list(eta_dict.keys())[-2], list(eta_dict.keys())[-1], 10, endpoint=True):
        eta_dict[eta] = calculate_reliability_from_mrr(beta=beta, eta=eta, mrr_year=mrr_year, gamma=gamma, quantity=quantity,
                                         mode=mode).at[mrr_year, 'MRR_' + mode]
    sorted_keys = list(eta_dict.keys())
    sorted_keys.sort()
    eta_dict = {i: eta_dict[i] for i in sorted_keys}

    eta_upper_bound = list({ k: v for k, v in eta_dict.items() if v < mrr_desired }.keys())[0]
    eta_lower_bound = list({ k: v for k, v in eta_dict.items() if v > mrr_desired }.keys())[-1]

    print(eta_upper_bound, eta_lower_bound)

    sensitivity = eta_upper_bound - eta_lower_bound
    print(sensitivity)

    optimization_dict = {
        sensitivity: .25*sensitivity,
        sensitivity * 0.5: .1*sensitivity,
        sensitivity * 0.1: .05*sensitivity,
        sensitivity * 0.01: 0.005*sensitivity,
        sensitivity * 0.001: 0.0001 * sensitivity,
        0: .1
    }
    eta = eta_lower_bound
    mrr_eval = 0
    past_eval = 100000
    mrr = []
    eta_range = []
    mrr_col = 'MRR_' + mode
    attempts = 0

    '''
    Notes:
        start by running initial guess - mrr_eval
        compare to mrr_desired
    '''
    while abs(mrr_eval - mrr_desired) / mrr_desired >= 0.001:
        output = calculate_reliability_from_mrr(beta=beta, eta=eta, mrr_year=mrr_year, gamma=gamma, quantity=quantity, mode=mode)
        mrr_eval = output.at[mrr_year, mrr_col]
        mrr.append(mrr_eval)
        eta_range.append(eta)
        pres_eval = abs(mrr_eval - mrr_desired) / mrr_desired * sensitivity
        print(f'Eta = {eta} | Eval Loop: {pres_eval} | ({mrr_eval} - {mrr_desired}) / {mrr_desired}')
        # print(f'present_eval: {pres_eval}')


        eta_modifier = [x for x in optimization_dict.keys() if x < pres_eval][0]
        print(f'eta_modifier: {eta_modifier}')
        eta += optimization_dict[eta_modifier]

        if past_eval < pres_eval:
            break
        past_eval = pres_eval

        attempts += 1
        if attempts > 100:
            break

    output = calculate_reliability_from_mrr(beta=beta, eta=eta, gamma=gamma, quantity=quantity, mode=mode)
    mrr.append(output.at[mrr_year,mrr_col])
    eta_range.append(eta)

    df = pd.DataFrame({'eta': eta_range, mrr_col: mrr})
    # print(df)
    df_closest = df.iloc[(df[mrr_col] - mrr_desired).abs().argsort()[:1]]

    if show_plot:
        print(df_closest['eta'].values[0])
        print(df_closest[mrr_col].values[0])
        mrr_plot = df.plot(x='eta', y=f'MRR_{mode}', marker='o', logy=True,
                           title='Eta selection for desired MRR target')
        mrr_plot.axhline(y=mrr_desired, color='r')
        mrr_plot.set_ylabel('MRR (log scale)')
        mrr_plot.text(x=min(eta_range), y=mrr_desired, s=mrr_desired, horizontalalignment='left',
                      verticalalignment='bottom')
        plt.show()

    # print(f'Optimized Eta: {df_closest['eta'].values[0]}')

    return df_closest['eta'].values[0]


# # Define the exponential decay function
# def exp_decay(x, a, b, c):
#     return a * np.exp(-b * x) + c



if __name__ == '__main__':
    # print('Reliability Target Conversion')
    #
    # print('Reliability Conversion')
    # rel = ReliabilityAtYear(reliability=0.9, beta=1.5)
    # print(rel)
    # print(rel.reliability_table)
    # print(rel.get_reliability_by_year())
    # print(rel.get_mrptv_at_year(year=3))
    #
    # print('MRpTV Conversion')
    # mrptv = MRpTVAtYear(mrptv=0.1, mrptv_year=3, beta=1.5, gamma=0, mode='at_year')
    # print(mrptv)
    # print(mrptv.reliability_table)


    # eta = calculate_eta_from_reliability(r_ts=0.9, beta=1.5, gamma=0, ts=10)
    # print(f'Eta: {eta} for reliability r_ts=0.9, beta=1.5 at ts=10')
    #
    # failure_rate_by_year = calculate_failure_rate_by_year(beta=1.5, eta=eta)
    # print('Failure Rate by Year')
    # print(failure_rate_by_year)
    #
    # unreliability_by_year = calculate_unreliability_by_year(beta=1.5, eta=eta)
    # print('Unreliability by Year')
    # print(unreliability_by_year)
    #
    # reliability_df = calculate_reliability_from_mrr(beta=1.5, eta=eta)
    # print('Reliability Summary from MRR')
    # print(reliability_df)
    #
    # reliability_at_year_6 = calculate_reliability_at_year_from_mrr(beta=1.5, eta=eta, year=6)
    # print(f'Reliability at Year 6: {reliability_at_year_6}')
    #
    # mrr_df = calculate_mrr_from_reliability(beta=1.5, eta=eta)
    # print('MRR Summary from Reliability')
    # print(mrr_df)
    #
    eta_from_mrr = eta_mrr_optimization_trial(mrr_desired=0.0001, mrr_year=3, beta=.5, show_plot=True)
    print(f'Optimized Eta: {eta_from_mrr}')
