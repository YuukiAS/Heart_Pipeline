"""
This script contains some utility functions for analyzing the extracted features.
"""

import pandas as pd
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import pingouin as pg
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector


def calculate_icc(data_1, data_2, eid_name_1, eid_name_2, features_dict):
    """
    Calculate ICC for each feature in the features_dict, where the key is the name in data_1
    and value is the name in data_2.
    """
    data_1 = data_1.sort_values(by=eid_name_1)
    data_2 = data_2.sort_values(by=eid_name_2)
    pd.set_option("display.width", 200)  # Make sure there is no line break for icc result
    eid_common = set(data_1[eid_name_1]).intersection(set(data_2[eid_name_2]))
    print(f"There are {len(eid_common)} common subjects in the two datasets.")
    for key, val in features_dict.items():
        print(f"Feature: {key} in data_1 and {val} in data_2")
        data_1_feature = data_1[data_1[eid_name_1].isin(eid_common)][key]
        data_2_feature = data_2[data_2[eid_name_2].isin(eid_common)][val]
        combined_df = pd.DataFrame(
            {
                "Subject": list(eid_common),
                "Rater1": pd.Series.tolist(data_1_feature),
                "Rater2": pd.Series.tolist(data_2_feature),
            }
        )
        combined_df = pd.melt(
            combined_df, id_vars="Subject", var_name="Rater", value_vars=["Rater1", "Rater2"], value_name="Value"
        )
        icc = pg.intraclass_corr(data=combined_df, targets="Subject", raters="Rater", ratings="Value")
        # https://pingouin-stats.org/build/html/generated/pingouin.intraclass_corr.html
        # We will use ICC3 and ICC3k as raters is a fixed set
        print(icc.iloc[[2, 5]])


def _bland_altman_plot(feature1, feature2, *args, **kwargs):
    mean = np.mean([feature1, feature2], axis=0)
    diff = feature1 - feature2
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, axis=0)

    plt.figure(figsize=(8, 6))
    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(mean_diff, color="gray", linestyle="--")
    plt.axhline(mean_diff + 1.96 * std_diff, color="gray", linestyle="--")
    plt.axhline(mean_diff - 1.96 * std_diff, color="gray", linestyle="--")
    plt.xlabel("Mean of two measurements")
    plt.ylabel("Difference between measurements")
    plt.title("Bland-Altman Plot")
    plt.grid(True)
    plt.show()


def bland_altman_plot(data1, data2, eid_name_1, eid_name_2, features_dict, *args, **kwargs):
    """
    Create a Bland-Altman plot for each feature in the features_dict,
    where the key is the name in data_1 and value is the name in data_2.
    """
    data1 = data1.sort_values(by=eid_name_1)
    data2 = data2.sort_values(by=eid_name_2)
    eid_common = set(data1[eid_name_1]).intersection(set(data2[eid_name_2]))
    print(f"There are {len(eid_common)} common subjects in the two datasets.")
    for key, val in features_dict.items():
        print(f"Feature: {key} in data_1 and {val} in data_2")
        data_1_feature = data1[data1[eid_name_1].isin(eid_common)][key]
        data_1_feature = np.array(data_1_feature)
        data_2_feature = data2[data2[eid_name_2].isin(eid_common)][val]
        data_2_feature = np.array(data_2_feature)
        _bland_altman_plot(data_1_feature, data_2_feature, *args, **kwargs)


def plot_time_series_double_x(
    x1, 
    x2, 
    y, 
    x1_label, 
    x2_label, 
    y_label, 
    x1_to_x2_func, 
    x2_to_x1_func, 
    title=None, 
    colors=None, 
    display=False
):
    """
    Plot time series data with two x-axis.

    Parameters:
    x1: List, x-axis 1 data.
    x2: List, x-axis 2 data.
    y: List, y-axis data.
    x1_label: String, label for x-axis 1.
    x2_label: String, label for x-axis 2.
    y_label: String, label for y-axis.
    colors: List, colors for the plot.
    x1_to_x2_func: Function, convert x1 to x2.
    x2_to_x1_func: Function, convert x2 to x1.
    """
    assert len(x1) == len(y)
    assert len(x2) == len(y)

    if colors is None:
        colors = ["b"] * len(y)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(x1, y)
    ax1.set_xlabel(x1_label)
    ax1.set_ylabel(y_label)
    if title is not None:
        ax1.set_title(title)

    ax2 = ax1.secondary_xaxis("top", functions=(x1_to_x2_func, x2_to_x1_func))
    ax2.set_xlabel(x2_label)

    plt.scatter(x1, y, c=colors)

    # if there is negative y, add a horizontal dotted line at y = 0
    if np.min(y) < 0:
        plt.axhline(y=0, color="gray", linestyle="--")

    if display:
        plt.show()

    return fig, ax1, ax2


def plot_time_series_double_x_y(
    x1,
    x2,
    y_plot,
    y_scatter,
    x1_label,
    x2_label,
    y_label,
    x1_to_x2_func,
    x2_to_x1_func,
    title=None,
    colors=None,
    display=False,
):
    """
    Plot time series data with two x-axis. Both scatter version and continuous version of y are supplied.

    Parameters:
    x1: List, x-axis 1 data.
    x2: List, x-axis 2 data.
    y_plot: List, y-axis data.
    y_scatter: List, y-axis data.
    x1_label: String, label for x-axis 1.
    x2_label: String, label for x-axis 2.
    y_label: String, label for y-axis.
    colors: List, colors for the plot.
    x1_to_x2_func: Function, convert x1 to x2.
    x2_to_x1_func: Function, convert x2 to x1.
    """
    assert len(x1) == len(y_plot)
    assert len(x2) == len(y_plot)
    assert len(y_scatter) == len(y_plot)

    if colors is None:
        colors = ["b"] * len(y_scatter)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(x1, y_plot)
    ax1.set_xlabel(x1_label)
    ax1.set_ylabel(y_label)
    if title is not None:
        ax1.set_title(title)

    ax2 = ax1.secondary_xaxis("top", functions=(x1_to_x2_func, x2_to_x1_func))
    ax2.set_xlabel(x2_label)

    plt.scatter(x1, y_scatter, c=colors)

    if np.min(y_scatter) < 0:
        plt.axhline(y=0, color="gray", linestyle="--")

    if display:
        plt.show()

    return fig, ax1, ax2


def analyze_time_series_derivative(x: list, y: list, n_pos, n_neg, method="ma"):
    """
    Analyze the derivative of a time series of features derived from cine MRI.
    We follow the approach described in https://pubmed.ncbi.nlm.nih.gov/28241751/ when deriving PER/PFR.
    """

    if n_pos not in [0, 1, 2] or n_neg not in [0, 1, 2]:
        raise ValueError("n_pos and n_neg must be smaller than 2")

    # peak rate
    PR_pos = []
    PR_neg = []
    T_PR_pos = []
    T_PR_neg = []

    # Note moving average (ma) will reduce the number of frames
    if method == "ma":
        x_ma = np.convolve(x, np.ones(3) / 3, mode="valid")
        y_ma = np.convolve(y, np.ones(3) / 3, mode="valid")

        y_ma_diff = np.gradient(y_ma, x_ma)

        peaks_pos = find_peaks(y_ma_diff, height=0)
        peaks_neg = find_peaks(-y_ma_diff, height=0)
        # define sort in descending order of values
        T_PR_pos = peaks_pos[0][np.argsort(-peaks_pos[1]["peak_heights"])[:n_pos]] if n_pos > 0 else []
        T_PR_neg = peaks_neg[0][np.argsort(-peaks_neg[1]["peak_heights"])[:n_neg]] if n_neg > 0 else []

        for T in T_PR_pos:
            if T < 2 or T > len(x_ma) - 2:
                raise IndexError("The peak is too close to the edge of the time series.")
            PR_pos.append((y_ma[T + 2] - y_ma[T - 2]) / (x_ma[T + 2] - x_ma[T - 2]))

        for T in T_PR_neg:
            if T < 1 or T > len(x_ma) - 2:
                raise IndexError("The peak is too close to the edge of the time series.")
            PR_neg.append((y_ma[T + 2] - y_ma[T - 1]) / (x_ma[T + 2] - x_ma[T - 1]))

    elif method == "loess":
        fANCOVA = importr("fANCOVA")
        x_r = FloatVector(x)
        y_r = FloatVector(y)
        loess_fit = fANCOVA.loess_as(x_r, y_r, degree=2, criterion="gcv")
        x_loess = np.array(loess_fit.rx2("x")).reshape(len(x),)
        y_loess = np.array(loess_fit.rx2("fitted"))

        y_diff = np.diff(y_loess) / np.diff(x_loess)

        peaks_pos = find_peaks(y_diff, height=0)
        peaks_neg = find_peaks(-y_diff, height=0)
        # define sort in descending order of values
        T_PR_pos = peaks_pos[0][np.argsort(-peaks_pos[1]["peak_heights"])[:n_pos]] if n_pos > 0 else []
        T_PR_neg = peaks_neg[0][np.argsort(-peaks_neg[1]["peak_heights"])[:n_neg]] if n_neg > 0 else []
        
        for T in T_PR_pos:
            if T < 2 or T > len(x_loess) - 2:
                raise IndexError("The peak is too close to the edge of the time series.")
            PR_pos.append(y_diff[T])
        
        for T in T_PR_neg:
            if T < 1 or T > len(x_loess) - 2:
                raise IndexError("The peak is too close to the edge of the time series.")
            PR_neg.append(y_diff[T])

    return T_PR_pos, T_PR_neg, PR_pos, PR_neg
