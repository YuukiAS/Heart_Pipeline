"""
This script contains some utility functions for analyzing the extracted features.
"""

import pandas as pd
import json5
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from collections import defaultdict

import pingouin as pg

import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from utils.log_utils import setup_logging

logger = setup_logging("analyze-utils")

# Install fANOCOVA package in R if not installed
if not rpackages.isinstalled("fANCOVA"):
    utils = rpackages.importr("utils")
    utils.chooseCRANmirror(ind=1)  # 选择CRAN镜像
    utils.install_packages("fANCOVA")


# Load analyze_utils.r for R functions
file_folder = os.path.dirname(os.path.abspath(__file__))
robjects.r["source"](os.path.join(file_folder, "analyze_utils.r"))

FEATURES_PER_ROW = 5  # If there are more than 10 features, we will splitu77 the boxplot


def calculate_icc(data_1, data_2, eid_name_1, eid_name_2, features_dict):
    """
    Calculate ICC for each feature in the features_dict, where the key is the name in data_1
    and value is the name in data_2.
    """
    data_1 = data_1.sort_values(by=eid_name_1)
    data_2 = data_2.sort_values(by=eid_name_2)
    pd.set_option("display.width", 200)  # Make sure there is no line break for icc result
    eid_common = set(data_1[eid_name_1]).intersection(set(data_2[eid_name_2]))
    logger.info(f"There are {len(eid_common)} common subjects in the two datasets.")
    for key, val in features_dict.items():
        logger.info(f"Feature: {key} in data_1 and {val} in data_2")
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
        logger.info(icc.iloc[[2, 5]])


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
    logger.info(f"There are {len(eid_common)} common subjects in the two datasets.")
    for key, val in features_dict.items():
        logger.info(f"Feature: {key} in data_1 and {val} in data_2")
        data_1_feature = data1[data1[eid_name_1].isin(eid_common)][key]
        data_1_feature = np.array(data_1_feature)
        data_2_feature = data2[data2[eid_name_2].isin(eid_common)][val]
        data_2_feature = np.array(data_2_feature)
        _bland_altman_plot(data_1_feature, data_2_feature, *args, **kwargs)


# There will be some variants of this function, such as allowing multiple y or stacked
def plot_time_series_dual_axes(
    x1, y, x1_label, x2_label, y_label, x1_to_x2_func, x2_to_x1_func, title=None, colors=None, display=False
):
    """
    Plot time series data with two x-axis.

    Parameters:
    x1: List, x-axis 1 data.
    y: List, y-axis data.
    x1_label: String, label for x-axis 1.
    x2_label: String, label for x-axis 2.
    y_label: String, label for y-axis.
    colors: List, colors for the plot.
    x1_to_x2_func: Function, convert x1 to x2.
    x2_to_x1_func: Function, convert x2 to x1.
    """
    assert len(x1) == len(y)

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


def plot_time_series_dual_axes_double_y(
    x1,
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
    Plot time series data with two x-axis and two y series.
    One time series will be visualized as line and another will be visualized as scatters.

    Parameters:
    x1: List, x-axis 1 data.
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


def plot_time_series_dual_axes_multiple_y(
    x1,
    y_list,
    legend_labels,
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
    Plot time series data with two x-axis and multiple y series.
    All time series will be visualized as lines.

    Parameters:
    x1: List, x-axis 1 data.
    y_list: List of lists, multiple y-axis data series.
    legend_labels: List of strings, labels for each y series in legend.
    x1_label: String, label for x-axis 1 (bottom).
    x2_label: String, label for x-axis 2 (top).
    y_label: String, label for shared y-axis.
    colors: List, colors for each y series.
    x1_to_x2_func: Function, convert x1 to x2.
    x2_to_x1_func: Function, convert x2 to x1.
    """
    for i, y in enumerate(y_list):
        assert len(x1) == len(y), f"x1 and y_list[{i}] must have the same length"
    assert len(y_list) == len(legend_labels), "y_list and legend_labels must have the same length"

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(y_list)))
    elif len(colors) < len(y_list):
        colors = colors * (len(y_list) // len(colors) + 1)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    for y, label, color in zip(y_list, legend_labels, colors):
        ax1.plot(x1, y, color=color, alpha=0.7, label=label)

        ax1.scatter(x1, y, color=color, edgecolors="k", s=50, zorder=3)

    ax1.set_xlabel(x1_label)
    ax1.set_ylabel(y_label)
    if title is not None:
        ax1.set_title(title)

    ax2 = ax1.secondary_xaxis("top", functions=(x1_to_x2_func, x2_to_x1_func))
    ax2.set_xlabel(x2_label)

    ax1.legend(loc="best")
    all_y = np.concatenate(y_list)
    ax1.set_ylim(np.min(all_y) - 0.1 * np.ptp(all_y), np.max(all_y) + 0.1 * np.ptp(all_y))
    if np.min(all_y) < 0:
        ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)

    if display:
        plt.show()

    return fig, ax1, ax2


def analyze_time_series_root(x: list, y: list, n_root=1, method="interpolate"):
    """
    Determine the location of the specified root of a time series of certain features derived from cine MRI.
    By default it will determine the location of the first root. The LOSS/LOWESS will be used to smooth the time series.
    Here x is usually the real time and y is the feature value.
    Two methods are supported: Round to nearest number and linear interpolation.
    """

    fANCOVA = importr("fANCOVA")
    x_r = FloatVector(x)
    y_r = FloatVector(y)
    loess_fit = fANCOVA.loess_as(x_r, y_r, degree=2, criterion="gcv")
    x_loess = np.array(loess_fit.rx2("x")).reshape(
        len(x),
    )
    y_loess = np.array(loess_fit.rx2("fitted"))

    if method == "interpolate":
        crossing_indices = np.where(np.diff(np.sign(y_loess)) != 0)[0]

        if len(crossing_indices) >= n_root:
            crossing_index = crossing_indices[n_root - 1]
            x1, x2 = x_loess[crossing_index], x_loess[crossing_index + 1]
            y1, y2 = y_loess[crossing_index], y_loess[crossing_index + 1]

            # linear interpolation
            root = x1 - y1 * (x2 - x1) / (y2 - y1)

            return root
        elif len(crossing_indices) == 0:
            raise ValueError("No root found in y.")
        else:
            raise ValueError("Not enough roots found in y.")

    elif method == "round":
        crossing_indices = np.where(np.diff(np.sign(y_loess)) != 0)[0]

        if len(crossing_indices) >= n_root:
            crossing_index = crossing_indices[n_root - 1]
            return round(x_loess[crossing_index])
        elif len(crossing_indices) == 0:
            raise ValueError("No root found in y.")
        else:
            raise ValueError("Not enough roots found in y.")
    else:
        raise ValueError("The method must be either interpolate or round.")


def analyze_time_series_derivative(x: list, y: list, n_pos, n_neg, method="slope"):
    """
    Analyze the peaks and times to peaks using derivative of the provided time series.
    Usually, x should be the real time and y should be the value of scalar feature.
    Two methods are supported:
    1. The slope method calculates the derivative through the slope of adjacent frames.
    2. The central difference method provided by NumPy.
    """

    if n_pos not in [0, 1, 2] or n_neg not in [0, 1, 2]:
        raise ValueError("n_pos and n_neg must be smaller than 2")

    # peak rate
    PR_pos = []
    PR_neg = []
    T_PR_pos = []
    T_PR_neg = []

    # Note this will reduce the number of frames
    if method == "slope":
        x_ma = np.convolve(x, np.ones(3) / 3, mode="valid")
        y_ma = np.convolve(y, np.ones(3) / 3, mode="valid")

        y_ma_diff = np.gradient(y_ma, x_ma)

        peaks_pos = find_peaks(y_ma_diff, height=0)
        peaks_neg = find_peaks(-y_ma_diff, height=0)
        # define sort in descending order of values
        T_PR_pos = peaks_pos[0][np.argsort(-peaks_pos[1]["peak_heights"])[:n_pos]] if n_pos > 0 else []
        T_PR_neg = peaks_neg[0][np.argsort(-peaks_neg[1]["peak_heights"])[:n_neg]] if n_neg > 0 else []

        # Ref Time-resolved tracking of the atrioventricular plane displacement in Cardiovascular Magnetic Resonance (CMR) images https://pubmed.ncbi.nlm.nih.gov/28241751/
        for T in T_PR_pos:  # usually for peak filling rate
            if T < 1 or T > len(x_ma) - 2:
                raise IndexError("The positive peak is too close to the boundary of the time series.")
            PR_pos.append((y_ma[T + 2] - y_ma[T - 1]) / (x_ma[T + 2] - x_ma[T - 1]))

        for T in T_PR_neg:  # usually for peak emptying rate
            if T < 2 or T > len(x_ma) - 2:
                raise IndexError("The negative peak is too close to the boundary of the time series.")
            PR_neg.append((y_ma[T + 2] - y_ma[T - 2]) / (x_ma[T + 2] - x_ma[T - 2]))

    elif method == "difference":
        # y_diff = np.diff(y_loess) / np.diff(x_loess)
        y_diff = np.gradient(y, x)

        peaks_pos = find_peaks(y_diff, height=0)
        peaks_neg = find_peaks(-y_diff, height=0)
        # define sort in descending order of values
        T_PR_pos = peaks_pos[0][np.argsort(-peaks_pos[1]["peak_heights"])[:n_pos]] if n_pos > 0 else []
        T_PR_neg = peaks_neg[0][np.argsort(-peaks_neg[1]["peak_heights"])[:n_neg]] if n_neg > 0 else []

        for T in T_PR_pos:
            # ensure consistency
            if T < 1 or T > len(x) - 2:
                raise IndexError("The positive peak is too close to the boundary of the time series.")
            PR_pos.append(y_diff[T])

        for T in T_PR_neg:
            if T < 2 or T > len(x) - 2:
                raise IndexError("The negative peak is too close to the boundary of the time series.")
            PR_neg.append(y_diff[T])

    else:
        raise ValueError("The method for analyzing must be either slope or difference.")

    return T_PR_pos, T_PR_neg, PR_pos, PR_neg


def calculate_reference_range(
    n, means=None, sds=None, lowers=None, uppers=None, method="frequentist", log_transform=True, diagnosis=False
):
    """
    Calculate the aggregated data reference range (AD RR) for a new individual based on the meta-analysis results.
    The CI for pooled mean and prediction for a new study mean (PI) are far narrower and
    do not capture healthy individual's full variation.
    Note: When the number of studies is small, the range of PI may still be wide.
    """
    if not n:
        raise ValueError("The number for each study must be provided.")
    if means is not None and sds is not None:
        means = means
        sds = sds
    elif lowers is not None and uppers is not None:
        means = [(lowers[i] + uppers[i]) / 2 for i in range(len(lowers))]
        # we use mean +- 2 * sd as the reference range
        sds = [(uppers[i] - lowers[i]) / 4 for i in range(len(lowers))]
    else:
        raise ValueError("Either means and sds or lower and upper bounds must be provided.")

    means_r = FloatVector(means)
    sds_r = FloatVector(sds)
    n_r = robjects.IntVector(n)
    # Ref Siegel et al. "A guide to estimating the reference range from a meta-analysis"
    if method == "frequentist":
        FreqFit = robjects.r["FreqFit"]
        result = FreqFit(means_r, sds_r, n_r)
    elif method == "empirical":
        EmpFit = robjects.r["EmpFit"]
        result = EmpFit(means_r, sds_r, n_r)
    else:
        raise ValueError("The method must be either frequentist or empirical.")

    CI = (round(result[0][0], 5), round(result[1][0], 5))  # define CI for pooled mean
    PI = (round(result[2][0], 5), round(result[3][0], 5))  # define PI for new study mean
    if np.any(np.array(means) <= 0):
        logger.warning("The study means contain non-positive values, log transformation is disabled.")
        log_transform = False
    if not log_transform:
        AD_RR = (round(result[4][0], 5), round(result[5][0], 5))  # define reference range for a new individual
    else:
        log_means = []
        log_sds = []
        # * We only log transform for reference range, not the new study mean
        # * Based on assumption that study means follow a normal distribuion with equal variance, also adjust for finite sample
        for mean, sd, sample_size in zip(means, sds, n):
            # logmean = log(mean / sqrt(1 + ((n - 1) / n) * (sd^2 / mean^2)))
            log_mean = np.log(mean / np.sqrt(1 + ((sample_size - 1) / sample_size) * (sd**2 / mean**2)))
            log_sd = np.sqrt(np.log(1 + ((sample_size - 1) / sample_size) * (sd**2 / mean**2)))

            log_means.append(log_mean)
            log_sds.append(log_sd)
        log_means_r = FloatVector(log_means)
        log_sds_r = FloatVector(log_sds)
        if method == "frequentist":
            FreqFit = robjects.r["FreqFit"]
            result_log = FreqFit(log_means_r, log_sds_r, n_r)
        elif method == "empirical":
            EmpFit = robjects.r["EmpFit"]
            result_log = EmpFit(log_means_r, log_sds_r, n_r)
        AD_RR = (round(np.exp(result_log[4][0]), 5), round(np.exp(result_log[5][0]), 5))

    if diagnosis:
        # * We will use QQ Plot to assess the normality assumption of the study means
        # * We will use a forest plot of SD to assess the equal variance assumption
        if not log_transform:
            QQPlot = robjects.r["QQPlot"]
            QQPlot(means_r)
            ForestPlot = robjects.r["ForestPlot"]
            ForestPlot(sds_r, n_r)
        else:
            QQPlot = robjects.r["QQPlot"]
            QQPlot(log_means_r)
            ForestPlot = robjects.r["ForestPlot"]
            ForestPlot(log_sds_r, n_r, log_transform=True)

        QQPlot_fig = plt.imread("qqplot.png")
        ForestPlot_fig = plt.imread("forestplot.png")
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(QQPlot_fig)
        plt.title("QQ Plot")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(ForestPlot_fig)
        plt.title("Forest Plot")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return CI, PI, AD_RR


def _query_reference_range(json_data, region, name, diagnosis=False):
    if region not in json_data:
        raise ValueError(f"Region {region} not found in the reference range json file.")
    for feature in json_data[region]:
        if feature["name"] == name:
            statistics = feature.get("reference_range")
            n_study = len(statistics)
            if n_study == 0:  # no reference range provided in JSON
                return None, None
            n = [statistic.get("n", None) for statistic in statistics]
            means = [statistic.get("mean", None) for statistic in statistics]
            sds = [statistic.get("sd", None) for statistic in statistics]
            lowers = [statistic.get("lower", None) for statistic in statistics]
            uppers = [statistic.get("upper", None) for statistic in statistics]
            if np.all(np.vectorize(lambda x: x is not None)(means)):
                # If all means and sds are not None
                if np.all(np.vectorize(lambda x: x is not None and x > 0)(lowers)):
                    CI, PI, AD_RR = calculate_reference_range(
                        n, lowers=lowers, uppers=uppers, log_transform=True, diagnosis=diagnosis
                    )
                else:
                    CI, PI, AD_RR = calculate_reference_range(n, means=means, sds=sds, log_transform=False, diagnosis=diagnosis)

            else:
                # Otherwise, convert lowers and uppers to means and sds
                for i in range(len(means)):
                    if means[i] is None:
                        means[i] = (lowers[i] + uppers[i]) / 2
                        sds[i] = (uppers[i] - lowers[i]) / 4
                # means1 = [(lowers[i] + uppers[i]) / 2 for i in range(n_study)]
                # sds1 = [(uppers[i] - lowers[i]) / 4 for i in range(n_study)]
                # means = [mean for mean in means if mean is not None] + means1
                # sds = [sd for sd in sds if sd is not None] + sds1
                if np.all(np.vectorize(lambda x: x > 0)(means)):
                    CI, PI, AD_RR = calculate_reference_range(n, means=means, sds=sds, log_transform=True, diagnosis=diagnosis)
                else:
                    CI, PI, AD_RR = calculate_reference_range(n, means=means, sds=sds, log_transform=False, diagnosis=diagnosis)

            # Correct PI when the mean is positive and sd is large
            if np.all(np.vectorize(lambda x: x is not None and x > 0)(means)):
                if PI[0] < 0:
                    PI = (0, PI[1])
            return CI, PI, AD_RR

    raise ValueError(f"Feature {name} in region {region} not found in the reference range json file.")


def analyze_aggregated_csv(
    csv_path: str,
    visualization_path: str,
    reference_range_json_path=None,
    specified_columns=None,
    specified_file_name=None,
    diagnosis=False,
):
    """
    The main function to analyze features extracted from the aggregated csv file.
    It will generate boxplot for numerical feature and pieplot for categorical feature.
    If the reference range json file is provided, it will also query the reference range for each feature.
    (This function may be further simplified in the future)
    """
    csv = pd.read_csv(csv_path)
    reference_range_json = None
    if reference_range_json_path:
        with open(reference_range_json_path, "r") as file:
            reference_range_json = json5.load(file)

    if specified_columns:
        # * We can manually specify the columns to be analyzed
        columns = specified_columns
        for column in columns:
            if csv[column].dtype in [object]:
                logger.warning("The specified column is not numeric, skipped.")
                columns.remove(column)
            # drop columns if specified
        csv = csv[columns]
    else:
        columns = csv.columns

    feature_pattern = r"([\w\s\-]+):\s+([\w\s\/\-\.\,]+(?:\([\w\s\-,\.]+\))?)\s*(?:\[([\w\/\^\%\-\d\.°]+)\])?"

    # * Option1: If we specify the columns, we will only generate plots for these columns
    if specified_columns:
        features_list = []  # we will report all specified features, regardless of region and unit
        if not specified_file_name:
            raise ValueError("If specified_columns is provided, specified_file_name must be provided.")
        for feature in specified_columns:
            match = re.match(feature_pattern, feature)
            if match:
                region = match.group(1)
                name = match.group(2).strip()
                unit = match.group(3) if match.group(3) else ""
                CI = None
                PI = None
                AD_RR = None
                if reference_range_json:
                    try:
                        CI, PI, AD_RR = _query_reference_range(reference_range_json, region, name, diagnosis=diagnosis)
                    except ValueError:
                        logger.warning(f"No reference range found for feature {name} in region {region}")
                features_list.append((region, name, unit, CI, PI, AD_RR))
            else:
                logger.warning(f"Feature {feature} does not match the naming convention, skipped.")
                continue

        _, names, units, CIs, PIs, AD_RRs = zip(*features_list)
        # Make sure units are the same
        if not all(unit == units[0] for unit in units):
            raise ValueError("All specified columns must have the same unit.")
        unit = units[0]
        reference_ranges = {name: (CI, PI, AD_RR) for name, CI, PI, AD_RR in zip(names, CIs, PIs, AD_RRs)}
        column_features = columns
        csv_features = csv[column_features]
        csv_melted = csv_features.melt(var_name="Feature", value_name=f"{unit}")
        csv_melted["Feature"] = csv_melted["Feature"].apply(lambda x: re.match(feature_pattern, x).group(2))
        feature_counts = csv_melted.groupby("Feature").size()
        feature_order = csv_melted.groupby("Feature")[f"{unit}"].mean().sort_values(ascending=False).index
        csv_melted["Feature"] = pd.Categorical(csv_melted["Feature"], categories=feature_order, ordered=True)
        palette = sns.color_palette("husl", len(csv_melted["Feature"].unique()))
        plt.figure(figsize=(20, 6))
        ax = sns.boxplot(x="Feature", y=f"{unit}", data=csv_melted, palette=palette, hue="Feature", legend=False)
        plt.xticks(fontsize=8)
        plt.xlabel("Feature", fontsize=10)
        for i, feature in enumerate(feature_order):
            count = feature_counts[feature]
            ax.text(i, ax.get_ylim()[1] * 0.95, f"(n={count})", ha="center", va="bottom", fontsize=10, color="black")
        plt.ylabel(f"{unit}", fontsize=10)
        if unit != "":
            plt.title(f"Boxplot of Features in with unit {unit}", fontsize=12)
        else:
            plt.title("Boxplot of Features in with no unit", fontsize=12)

        # Add reference ranges to the boxplot
        reference_ranges_ordered = []
        for name_i in feature_order:
            name_str = str(name_i).strip()
            if reference_ranges[name_str]:
                reference_ranges_ordered.append(reference_ranges[name_str])
            else:
                reference_ranges_ordered.append((None, None, None))
        for index, reference_range in enumerate(reference_ranges_ordered):
            # If there is reference range for certain features, add them to the boxplot
            CI, PI, AD_RR = reference_range
            if AD_RR:
                ax.fill_betweenx(
                    y=[AD_RR[0], AD_RR[1]],
                    x1=index - 0.4,
                    x2=index + 0.4,
                    color="lightgreen",
                    alpha=0.2,
                    label="AD RR for new individual",
                    zorder=10,
                )
                ax.hlines(
                    y=AD_RR[0], xmin=index - 0.4, xmax=index + 0.4, color="lightgreen", alpha=0.5, linestyle="--", zorder=10
                )
                ax.hlines(
                    y=AD_RR[1], xmin=index - 0.4, xmax=index + 0.4, color="lightgreen", alpha=0.5, linestyle="--", zorder=10
                )
            if PI:
                ax.fill_betweenx(
                    y=[PI[0], PI[1]],
                    x1=index - 0.4,
                    x2=index + 0.4,
                    color="lightblue",
                    alpha=0.2,
                    label="PI for new study mean",
                    zorder=10,
                )
                ax.hlines(y=PI[0], xmin=index - 0.4, xmax=index + 0.4, color="lightblue", alpha=0.5, linestyle="--", zorder=10)
                ax.hlines(y=PI[1], xmin=index - 0.4, xmax=index + 0.4, color="lightblue", alpha=0.5, linestyle="--", zorder=10)
            if CI:
                ax.fill_betweenx(
                    y=[CI[0], CI[1]],
                    x1=index - 0.4,
                    x2=index + 0.4,
                    color="lightpink",
                    alpha=0.2,
                    label="CI for pooled mean",
                    zorder=10,
                )
                ax.hlines(y=CI[0], xmin=index - 0.4, xmax=index + 0.4, color="lightpink", alpha=0.5, linestyle="--", zorder=10)
                ax.hlines(y=CI[1], xmin=index - 0.4, xmax=index + 0.4, color="lightpink", alpha=0.5, linestyle="--", zorder=10)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="upper right")
        plt.savefig(f"{visualization_path}/{specified_file_name}.png")
        plt.close()
        return

    # * Option2: Otherwise we will generate a few boxplots, each figure corresponds to features in the same region with same unit
    else:
        features_dict = defaultdict(list)  # define The dictionary to hold the features for each region
        for feature in columns:
            if feature == "eid" or feature == "Unnamed: 0":
                continue
            match = re.match(feature_pattern, feature)
            if match:
                region = match.group(1)
                name = match.group(2).strip()
                unit = match.group(3) if match.group(3) else ""  # for some features thre is no unit
                # print(f"Region: {region}, Name: {name}, Unit: {unit}")
                CI = None
                PI = None
                AD_RR = None
                if reference_range_json and csv[feature].dtype not in [object]:
                    # * We only consider reference range for numeric features
                    try:
                        CI, PI, AD_RR = _query_reference_range(reference_range_json, region, name, diagnosis=diagnosis)
                    except ValueError:
                        logger.warning(f"No reference range found for feature {name} in region {region}")

                features_dict[(region, unit)].append((name, CI, PI, AD_RR))
            else:
                logger.warning(f"Feature {feature} does not match the naming convention, skipped.")
                continue

        cnt = 0
        region_old = None
        for (region, unit), features_info in features_dict.items():
            features, CIs, PIs, AD_RRs = zip(*features_info)
            # this will be extended and sorted later
            reference_ranges = {feature: (CI, PI, AD_RR) for feature, CI, PI, AD_RR in zip(features, CIs, PIs, AD_RRs)}
            if region != region_old:
                cnt = 0
            region_old = region
            cnt += 1
            # * For most numerical features with units, recover the column names
            if unit != "":
                logger.info(f"Generating boxplots for all features in {region} with unit {unit}")
                columns_features = [f"{region}: {feature} [{unit}]" for feature in features]
            # * Note some features can be categorical or have no unit. We generate pieplot for categorical features
            else:
                columns_features = [f"{region}: {feature}" for feature in features]

                columns_features_numeric = [
                    column_feature for column_feature in columns_features if csv[column_feature].dtype not in [object]
                ]
                columns_features_categorical = [
                    column_feature for column_feature in columns_features if csv[column_feature].dtype in [object]
                ]
                # Generate pieplot
                for column_feature_categorical in columns_features_categorical:
                    feature = column_feature_categorical.split(": ")[1]  # name of feature
                    logger.info(f"Generating pieplot for categorical feature {feature}")
                    feature_categorical = csv[column_feature_categorical]
                    feature_counts = feature_categorical.value_counts()
                    palette = sns.color_palette("husl", len(feature_counts))
                    _, ax = plt.subplots(figsize=(10, 6))
                    wedges, _, _ = ax.pie(
                        feature_counts.values,
                        labels=None,
                        autopct="",
                        startangle=90,
                        colors=palette,
                    )
                    for i, wedge in enumerate(wedges):
                        angle = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1
                        x = np.cos(np.radians(angle))
                        y = np.sin(np.radians(angle))

                        if feature_counts.values[i] / feature_counts.sum() < 0.05:
                            ax.annotate(
                                f"{feature_counts.values[i] / feature_counts.sum() * 100:.1f}%",
                                xy=(x * 0.6, y * 0.6),
                                xytext=(x * 1.2, y * 1.2),
                                arrowprops=dict(arrowstyle="->", color="black"),
                                ha="center",
                                fontsize=8,
                            )
                        else:
                            ax.text(
                                x * 0.6,
                                y * 0.6,
                                f"{feature_counts.values[i] / feature_counts.sum() * 100:.1f}%",
                                ha="center",
                                fontsize=8,
                            )
                    # avoid possible overlap
                    ax.legend(
                        labels=feature_counts.index,
                        loc="lower right",
                        bbox_to_anchor=(1, 0, 0.5, 1),
                    )
                    plt.title(f"Pieplot of categorical feature {feature}")
                    plt.tight_layout(rect=[0, 0, 1, 0.95])
                    plt.savefig(f"{visualization_path}/{region}_{feature}.png")
                    plt.close()

                # The remaining numerical features with
                columns_features = columns_features_numeric
                if len(columns_features) == 0:
                    # no boxplot generated if all these features are categorical and there is no feature without unit
                    cnt -= 1
                    continue
                else:
                    logger.info(f"Generating boxplots for all features in {region} with no unit")

            if len(columns_features) > FEATURES_PER_ROW:
                # * If there are too many features with same unit, we will generate separete boxplots
                n_boxplots = len(columns_features) // FEATURES_PER_ROW + 1
                for i in range(n_boxplots):
                    columns_features_i = columns_features[
                        i * FEATURES_PER_ROW : min((i + 1) * FEATURES_PER_ROW, len(columns_features))
                    ]
                    analyze_aggregated_csv(
                        csv_path=csv_path,
                        visualization_path=visualization_path,
                        reference_range_json_path=reference_range_json_path,
                        specified_columns=columns_features_i,
                        specified_file_name=f"{region}_{cnt}_{i + 1}",
                    )
            else:
                csv_features = csv[columns_features]
                csv_melted = csv_features.melt(var_name="Feature", value_name=f"{unit}")
                # replace the long name with short name
                csv_melted["Feature"] = csv_melted["Feature"].apply(lambda x: re.match(feature_pattern, x).group(2))
                # coun the number of subjects for each feature
                feature_counts = csv_melted.groupby("Feature").size()
                # sort the features by the mean value in descending order
                feature_order = csv_melted.groupby("Feature")[f"{unit}"].mean().sort_values(ascending=False).index
                csv_melted["Feature"] = pd.Categorical(csv_melted["Feature"], categories=feature_order, ordered=True)
                # * Generate boxplot
                palette = sns.color_palette("husl", len(csv_melted["Feature"].unique()))
                plt.figure(figsize=(15, 6))
                ax = sns.boxplot(x="Feature", y=f"{unit}", data=csv_melted, palette=palette, hue="Feature", legend=False)
                plt.xticks(fontsize=8)
                plt.xlabel("Feature", fontsize=10)
                for i, feature in enumerate(feature_order):
                    count = feature_counts[feature]
                    ax.text(i, ax.get_ylim()[1] * 0.95, f"(n={count})", ha="center", va="bottom", fontsize=10, color="black")
                plt.ylabel(f"{unit}", fontsize=10)
                if unit != "":
                    plt.title(f"Boxplot of Features in {region} with unit {unit}", fontsize=12)
                else:
                    plt.title(f"Boxplot of Features in {region} with no unit", fontsize=12)

                # If there is reference range for certain features, add them to the boxplot
                # We need to sort according to the feature_order before adding reference range
                reference_ranges_ordered = []
                for name_i in feature_order:
                    name_str = str(name_i).strip()
                    if reference_ranges[name_str]:
                        reference_ranges_ordered.append(reference_ranges[name_str])
                    else:
                        reference_ranges_ordered.append((None, None, None))
                    # for index, reference_range in enumerate(reference_ranges):
                for index, reference_range in enumerate(reference_ranges_ordered):
                    CI, PI, AD_RR = reference_range
                    if AD_RR:
                        ax.fill_betweenx(
                            y=[AD_RR[0], AD_RR[1]],
                            x1=index - 0.4,
                            x2=index + 0.4,
                            color="lightgreen",
                            alpha=0.2,
                            label="Reference Range for new individual",
                            zorder=10,
                        )
                        ax.hlines(
                            y=AD_RR[0],
                            xmin=index - 0.4,
                            xmax=index + 0.4,
                            color="lightgreen",
                            alpha=0.5,
                            linestyle="--",
                            zorder=10,
                        )
                        ax.hlines(
                            y=AD_RR[1],
                            xmin=index - 0.4,
                            xmax=index + 0.4,
                            color="lightgreen",
                            alpha=0.5,
                            linestyle="--",
                            zorder=10,
                        )
                    if PI:
                        ax.fill_betweenx(
                            y=[PI[0], PI[1]],
                            x1=index - 0.4,
                            x2=index + 0.4,
                            color="lightblue",
                            alpha=0.2,
                            label="PI for new study mean",
                            zorder=10,
                        )
                        ax.hlines(
                            y=PI[0], xmin=index - 0.4, xmax=index + 0.4, color="lightblue", alpha=0.6, linestyle="--", zorder=10
                        )
                        ax.hlines(
                            y=PI[1], xmin=index - 0.4, xmax=index + 0.4, color="lightblue", alpha=0.6, linestyle="--", zorder=10
                        )
                    if CI:
                        ax.fill_betweenx(
                            y=[CI[0], CI[1]],
                            x1=index - 0.4,
                            x2=index + 0.4,
                            color="lightpink",
                            alpha=0.2,
                            label="CI for pooled mean",
                            zorder=10,
                        )
                        ax.hlines(
                            y=CI[0], xmin=index - 0.4, xmax=index + 0.4, color="lightpink", alpha=0.7, linestyle="--", zorder=10
                        )
                        ax.hlines(
                            y=CI[1], xmin=index - 0.4, xmax=index + 0.4, color="lightpink", alpha=0.7, linestyle="--", zorder=10
                        )
                    # if CI or PI or AD_RR:
                    # plt.legend(loc="upper right")
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                # remove duplicate labels
                handles, labels = ax.get_legend_handles_labels()
                unique = dict(zip(labels, handles))
                # since we sort descendingly, legend is better put at the top
                ax.legend(unique.values(), unique.keys(), loc="upper right")
                plt.savefig(f"{visualization_path}/{region}_{cnt}.png")
                plt.close()


def plot_bulls_eye(data, title=None, label=None, vmin=None, vmax=None, cmap="Reds", color_line="black"):
    """
    Plot the bull's eye plot.
    For an example of Bull's eye plot, refer to https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4862218/pdf/40001_2016_Article_216.pdf.

    data: values for 16 segments
    vmin: minimum value for the colormap, values smaller than vmin will be colored as the minimum value
    vmax: maximum value for the colormap, values larger than vmax will be colored as the maximum value
    """
    if len(data) != 16:
        logger.error("The length of data must be 16, corresponding to each AHA segment.")
        exit(1)

    # check that data must have same sign
    if not (np.all(data >= 0) or np.all(data <= 0)):
        raise ValueError("The data must have the same sign.")
    # If all data are negative, we will cast them to positive for visualization
    data_text = data
    if np.all(data <= 0):
        data = -data

    if vmin is None:
        vmin = min(data)
    if vmax is None:
        vmax = max(data)

    fig, ax = plt.subplots(figsize=(6, 6))

    # The cartesian coordinate and the polar coordinate
    x = np.linspace(-1, 1, 201)
    y = np.linspace(-1, 1, 201)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx * xx + yy * yy)
    theta = np.degrees(np.arctan2(yy, xx))

    # The radius and degree for each segment
    R1, R2, R3, R4 = 1, 0.65, 0.3, 0.0
    rad_deg = {
        # * By default, it will go clockwise (0 degree is on the right, 90 degree is on the bottom)
        # * However, as we set ax.invert_yaxis(), it will now go counter-clockwise (90 degree is on the top)
        # Outer circle
        1: (R1, R2, 60, 120),  # from 60 degree to 120 degree
        2: (R1, R2, 120, 180),
        3: (R1, R2, -180, -120),
        4: (R1, R2, -120, -60),
        5: (R1, R2, -60, 0),
        6: (R1, R2, 0, 60),
        # Middle circle
        7: (R2, R3, 60, 120),
        8: (R2, R3, 120, 180),
        9: (R2, R3, -180, -120),
        10: (R2, R3, -120, -60),
        11: (R2, R3, -60, 0),
        12: (R2, R3, 0, 60),
        # Inner circle
        13: (R3, R4, 45, 135),
        14: (R3, R4, 135, -135),
        15: (R3, R4, -135, -45),
        16: (R3, R4, -45, 45),
    }

    # Plot the segments
    canvas = np.zeros(xx.shape)
    cx, cy = (np.array(xx.shape) - 1) / 2
    sz = cx  # the size of the canvas (maximum radius)

    for i in range(1, 17):
        # Add colors to the plot
        val = data[i - 1]
        r1, r2, theta1, theta2 = rad_deg[i]
        if i != 14 and i != 16:
            # select region in canvas and assign value for color
            mask = ((r < r1) & (r >= r2)) & ((theta >= theta1) & (theta < theta2))
        else:
            # Special case for segment 14 and 16, which cross the x-axis
            if i == 14:
                # theta1 = 135, theta2 = -135
                mask1 = ((r < r1) & (r >= r2)) & ((theta >= theta1) & (theta <= 180))
                mask2 = ((r < r1) & (r >= r2)) & ((theta >= -180) & (theta < theta2))
            else:
                # theta1 = -45, theta2= 45            
                mask1 = ((r < r1) & (r >= r2)) & ((theta >= theta1) & (theta < 0))
                mask2 = ((r < r1) & (r >= r2)) & ((theta >= 0) & (theta < theta2))                
            mask = mask1 | mask2  # take the union of two sub-mask to get the complete mask
        canvas[mask] = val

        # Add texts to the plot
        val_text = data_text[i - 1]
        mid_r = (r1 + r2) / 2
        mid_theta = (theta1 + theta2) / 2
        # Special case for segment 14 and 16
        if i == 14:
            mid_theta = -180
        elif i == 16:
            mid_theta = 0
        text_x = cx + sz * mid_r * np.cos(np.radians(mid_theta))
        text_y = cy + sz * mid_r * np.sin(np.radians(mid_theta))
        ax.text(text_x, text_y, f"{val_text:.2f}", color="black", fontsize=10, ha="center", va="center")

    im = ax.imshow(canvas, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=fig.axes[0])
    if label:
        cbar.set_label(label)
    if title:
        ax.set_title(title)
    ax.axis("off")
    ax.invert_yaxis()

    # Plot the circles
    for r in [R1, R2, R3]:
        deg = np.linspace(0, 2 * np.pi, 201)
        circle_x = cx + sz * r * np.cos(deg)
        circle_y = cy + sz * r * np.sin(deg)
        ax.plot(circle_x, circle_y, color=color_line)

    # Plot the lines between segments
    for i in range(1, 17):
        r1, r2, theta1, theta2 = rad_deg[i]
        line_x = cx + sz * np.array([r1, r2]) * np.cos(np.radians(theta1))
        line_y = cy + sz * np.array([r1, r2]) * np.sin(np.radians(theta1))
        ax.plot(line_x, line_y, color=color_line)

    # Plot the indicator for RV insertion points
    for i in [2, 4]:
        r1, r2, theta1, theta2 = rad_deg[i]
        x = cx + sz * r1 * np.cos(np.radians(theta1))
        y = cy + sz * r1 * np.sin(np.radians(theta1))
        # y = cy + sz * r1 * np.sin(np.radians(0))
        ax.plot([x, x - sz * 0.2], [y, y], color=color_line, linestyle="--")
        ax.annotate(
            "RV Intersection", 
            xy=(x, y), 
            xytext=(x - sz * 0.4, y + np.where(y >= 100, 1, -1) * sz * 0.2),
            arrowprops=dict(facecolor='black', arrowstyle='->'),
            fontsize=8,
            color=color_line
        )
    return fig, ax