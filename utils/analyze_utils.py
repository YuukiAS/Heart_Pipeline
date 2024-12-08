"""
This script contains some utility functions for analyzing the extracted features.
"""

import pandas as pd
import json
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from collections import defaultdict
import pingouin as pg
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from utils.log_utils import setup_logging

logger = setup_logging("analyze-utils")

# Load analyze_utils.r for R functions
file_folder = os.path.dirname(os.path.abspath(__file__))
robjects.r["source"](os.path.join(file_folder, "analyze_utils.r"))


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


def plot_time_series_double_x(
    x1, x2, y, x1_label, x2_label, y_label, x1_to_x2_func, x2_to_x1_func, title=None, colors=None, display=False
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
        x_loess = np.array(loess_fit.rx2("x")).reshape(
            len(x),
        )
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
        means = [(lowers[i] + uppers[i]) / 2 for i in range(n)]
        # we use mean +- 2 * sd as the reference range
        sds = [(uppers[i] - lowers[i]) / 4 for i in range(n)]
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

    PI = (round(result[0][0], 5), round(result[1][0], 5))  # define PI for new study mean
    if np.any(np.array(means) <= 0):
        logger.warning("The study means contain non-positive values, log transformation is disabled.")
        log_transform = False
    if not log_transform:
        AD_RR = (round(result[2][0], 5), round(result[3][0], 5))  # define reference range for a new individual
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
        AD_RR = (round(np.exp(result_log[2][0]), 5), round(np.exp(result_log[3][0]), 5))

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

    return PI, AD_RR


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
                # if all means and sds are not None
                if np.all(np.vectorize(lambda x: x is not None and x > 0)(lowers)):
                    PI, AD_RR = calculate_reference_range(
                        n, lowers=lowers, uppers=uppers, log_transform=True, diagnosis=diagnosis
                    )
                else:
                    PI, AD_RR = calculate_reference_range(n, means=means, sds=sds, log_transform=False, diagnosis=diagnosis)

            else:
                # convert lowers and uppers to means and sds
                means1 = [(lowers[i] + uppers[i]) / 2 for i in range(n_study)]
                sds1 = [(uppers[i] - lowers[i]) / 4 for i in range(n_study)]
                # concatenate the two lists
                means = [mean for mean in means if mean is not None] + means1
                sds = [sd for sd in sds if sd is not None] + sds1
                if np.all(np.vectorize(lambda x: x > 0)(means)):
                    PI, AD_RR = calculate_reference_range(n, means=means, sds=sds, log_transform=True, diagnosis=diagnosis)
                else:
                    PI, AD_RR = calculate_reference_range(n, means=means, sds=sds, log_transform=False, diagnosis=diagnosis)

            return PI, AD_RR

    raise ValueError(f"Feature {name} in region {region} not found in the reference range json file.")


def analyze_aggregated_csv(
    csv_path: str, visualization_path: str, reference_range_json_path=None, specified_columns=None, diagnosis=False
):
    """
    The main function to analyze features extracted from the aggregated csv file.
    It will generate boxplot for numerical feature and pieplot for categorical feature.
    If the reference range json file is provided, it will also query the reference range for each feature.
    """
    csv = pd.read_csv(csv_path)
    reference_range_json = None
    if reference_range_json_path:
        with open(reference_range_json_path, "r") as file:
            reference_range_json = json.load(file)

    if specified_columns:
        # * We can manually specify the columns to be analyzed
        columns = specified_columns
        # drop columns if specified
        csv = csv[columns]
    else:
        columns = csv.columns

    feature_pattern = r"([\w\s\-]+):\s+([\w\s\/\-\.\,]+(?:\([\w\s\-,\.]+\))?)\s*(?:\[([\w\/\^\%\-\d\.°]+)\])?"
    features_dict = defaultdict(list)  # define The dictionary to hold the features for each region

    # * Option1: If we specify the columns, we will only generate plots for these columns
    if specified_columns:
        # todo
        pass

    # * Option2: Otherwise we will generate a few boxplots, each figure corresponds to features in the same region with same unit
    else:
        for feature in columns:
            if feature == "eid" or feature == "Unnamed: 0":
                continue
            match = re.match(feature_pattern, feature)
            if match:
                region = match.group(1)
                name = match.group(2).strip()
                unit = match.group(3) if match.group(3) else ""  # for some features thre is no unit
                # print(f"Region: {region}, Name: {name}, Unit: {unit}")
                PI = None
                AD_RR = None
                if reference_range_json:
                    try:
                        PI, AD_RR = _query_reference_range(reference_range_json, region, name, diagnosis=diagnosis)
                    except ValueError:
                        logger.warning(f"No reference range found for feature {name} in region {region}")

                features_dict[(region, unit)].append((name, PI, AD_RR))

            else:
                logger.warning(f"Feature {feature} does not match the naming convention, skipped.")
                continue

        cnt = 0
        region_old = None
        for (region, unit), features_info in features_dict.items():
            features, PIs, AD_RRs = zip(*features_info)
            reference_ranges = list(zip(PIs, AD_RRs))
            if region != region_old:
                cnt = 0
            region_old = region
            cnt += 1
            # recover the column names
            if unit != "":
                logger.info(f"Generating boxplots for all features in {region} with unit {unit}")
                columns_features = [f"{region}: {feature} [{unit}]" for feature in features]
            else:
                columns_features = [f"{region}: {feature}" for feature in features]
                # * Note some features can be categorical, in this case we generate separate pieplot for each of them
                columns_features_numeric = [
                    column_feature for column_feature in columns_features if csv[column_feature].dtype not in [object]
                ]
                columns_features_categorical = [
                    column_feature for column_feature in columns_features if csv[column_feature].dtype in [object]
                ]
                # * Generate pieplot
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

                columns_features = columns_features_numeric
                if not columns_features == 0:
                    cnt -= 1
                    continue
                else:
                    logger.info(f"Generating boxplots for all features in {region} with no unit")

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
            for index, reference_range in enumerate(reference_ranges):
                PI, AD_RR = reference_range
                if PI:
                    ax.fill_betweenx(
                        y=[PI[0], PI[1]],
                        x1=index - 0.4, 
                        x2=index + 0.4,
                        color="lightgreen",
                        alpha=0.3,
                        label="PI for new study mean",
                    )
                if AD_RR:
                    ax.fill_betweenx(
                        y=[AD_RR[0], AD_RR[1]],
                        x1=index - 0.4, 
                        x2=index + 0.4,
                        color="lightblue",
                        alpha=0.3,
                        label="Reference Range for new individual",
                    )
                if PI or AD_RR:
                    plt.legend(loc="lower right")
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            plt.savefig(f"{visualization_path}/{region}_{cnt}.png")
            plt.close()


# Matplotlib

# fig, ax = plt.subplots(figsize=(8, 6))
# box = ax.boxplot(data, patch_artist=True, boxprops=dict(facecolor="lightblue"))

# ax.axhspan(ref_min, ref_max, color='lightgreen', alpha=0.3, label='Reference Range')

# ax.set_xticklabels(['Group 1', 'Group 2', 'Group 3'])
# ax.set_ylabel('Values')
# ax.legend()

# plt.show()

# sns.set(style="whitegrid")

# fig, ax = plt.subplots(figsize=(8, 6))
# sns.boxplot(x="Group", y="Value", data=df, palette="pastel", ax=ax)

# ax.axhspan(ref_min, ref_max, color='lightgreen', alpha=0.3, label='Reference Range')

# ax.set_ylabel('Values')
# ax.legend()

# plt.show()
