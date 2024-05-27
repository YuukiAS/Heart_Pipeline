"""
This script contains some utility functions for analyzing the extracted features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pingouin as pg

def calculate_icc(data_1, data_2, eid_name_1, eid_name_2, features_dict):
    """
    Calculate ICC for each feature in the features_dict, where the key is the name in data_1
    and value is the name in data_2.
    """
    data_1 = data_1.sort_values(by=eid_name_1)
    data_2 = data_2.sort_values(by=eid_name_2)
    pd.set_option('display.width', 200)  # Make sure there is no line break for icc result
    eid_common = set(data_1[eid_name_1]).intersection(set(data_2[eid_name_2]))
    print(f"There are {len(eid_common)} common subjects in the two datasets.")
    for key, val in features_dict.items():
        print(f"Feature: {key} in data_1 and {val} in data_2")
        data_1_feature = data_1[data_1[eid_name_1].isin(eid_common)][key]
        data_2_feature = data_2[data_2[eid_name_2].isin(eid_common)][val]
        combined_df = pd.DataFrame({
            'Subject': list(eid_common),
            'Rater1': pd.Series.tolist(data_1_feature),
            'Rater2': pd.Series.tolist(data_2_feature)
        })
        combined_df = pd.melt(combined_df, id_vars='Subject', var_name='Rater', 
                                           value_vars=['Rater1', 'Rater2'], value_name='Value')
        icc = pg.intraclass_corr(data=combined_df, targets='Subject', raters='Rater', ratings='Value')
        # https://pingouin-stats.org/build/html/generated/pingouin.intraclass_corr.html
        # We will use ICC3 and ICC3k as raters is a fixed set
        print(icc.iloc[[2,5]])


def _bland_altman_plot(feature1, feature2, *args, **kwargs):
    mean = np.mean([feature1, feature2], axis=0)
    diff = feature1 - feature2
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, axis=0)

    plt.figure(figsize=(8, 6))
    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(mean_diff, color='gray', linestyle='--')
    plt.axhline(mean_diff + 1.96 * std_diff, color='gray', linestyle='--')
    plt.axhline(mean_diff - 1.96 * std_diff, color='gray', linestyle='--')
    plt.xlabel('Mean of two measurements')
    plt.ylabel('Difference between measurements')
    plt.title('Bland-Altman Plot')
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