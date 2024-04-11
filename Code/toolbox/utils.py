# =============================================================================
# Toolbox by Katharina Lingelbach
# =============================================================================
import sys
import os
import numpy as np
import random
import scipy.stats as stats


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def find_nearest_idx(array: list, value: float) -> int:
    """
    Get nearest index by value

    Parameters
    ----------
    array: list like array with values to be searched
    value: float value to search in the array

    Returns
    -------
    index of the nearest value in the array

    """
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def get_sampling_rate(timestamp):
    """

    Estimates the sampling rate from the data timestamp
    Parameters
    ----------
    timestamp: 1d numpy array with timestamps

    Returns
    -------
    sampling rate
    """
    fs = 0
    for i in range(len(timestamp) - 1):
        fs += timestamp[i + 1] - timestamp[i]

    # fs: sampling interval (every X ms)
    fs = fs / (len(timestamp) - 1)

    # sampling rate in Hz (Samples per Second)
    if fs > 0:
        sr = round(1000 / fs, 2)
    else:
        sr = 0

    return sr, fs


def remove_empty_tuples(tuples):
    dropped = []
    reason = []
    for i, t in enumerate(tuples):
        if t != ():
            dropped.append(i)
            reason.append(t[0])
    return dropped, reason


def percentiles(lst_vals, alpha, func='mean'):
    lower = np.percentile(np.array(lst_vals), ((1.0 - alpha) / 2.0) * 100, axis=0)
    upper = np.percentile(lst_vals, (alpha + ((1.0 - alpha) / 2.0)) * 100, axis=0)
    if func == 'mean':
        mean = np.mean(lst_vals, axis=0)
    elif func == 'median':
        mean = np.median(lst_vals, axis=0)
    return lower, mean, upper


def bootstrapping(input_sample,
                  sample_size=None,
                  numb_iterations=1000,
                  alpha=0.95,
                  plot_hist=False,
                  as_dict=True,
                  func='mean'):  # mean, median

    if sample_size == None:
        sample_size = len(input_sample)

    lst_means = []

    # Bootstrapping
    for i in range(numb_iterations):
        try:
            re_sampled = random.choices(input_sample.values, k=sample_size)
        except:
            re_sampled = random.choices(input_sample, k=sample_size)

        if func == 'mean':
            lst_means.append(np.nanmean(np.array(re_sampled), axis=0))
        elif func == 'median':
            lst_means.append(np.median(np.array(re_sampled), axis=0))
        # lst_means.append(np.median(np.array(re_sampled), axis=0))

    # CI
    lower, mean, upper = percentiles(lst_means, alpha)
    dict_return = {'lower': lower, 'mean': mean, 'upper': upper}

    if as_dict:
        return dict_return
    else:
        return mean, np.array([np.abs(lower - mean), (upper - mean)])


def conf_limits_ncf(F, df_effect, df_error, conf_level=0.90):
    alpha = 1 - conf_level
    crit_value = stats.f.ppf(1 - alpha / 2, df_effect, df_error)
    lower_limit = F / crit_value
    upper_limit = F * crit_value
    return lower_limit, upper_limit


def partial_eta2_from_F(lmbda, df_effect, df_error):
    partial_eta2 = lmbda / (lmbda + df_effect + df_error + 1)
    return partial_eta2


def partial_eta_squared(F, df_effect, df_error):
    partial_eta2 = partial_eta2_from_F(F, df_effect, df_error)
    return partial_eta2


def partial_eta_squared_ci(F, df_effect, df_error, conf_level=0.90):
    F_limits = conf_limits_ncf(F, df_effect, df_error, conf_level=conf_level)

    LL_partial_eta2 = partial_eta2_from_F(F_limits[0], df_effect, df_error)
    UL_partial_eta2 = partial_eta2_from_F(F_limits[1], df_effect, df_error)

    return_string = f"[.{str(round(LL_partial_eta2, 2)).split('.')[1]}, .{str(round(UL_partial_eta2, 2)).split('.')[1]}]"
    return return_string
