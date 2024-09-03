"""Compare the persistence of weather regimes leading to events to random weather regimes that not necessarily lead to events.
Output saved, and plotted together with the comparison of strength in 03c_strength_regime_events.py

Author: Bram van Duinen (bramvduinen@gmail.com)

"""

# %%
# IMPORTS
import os

import numpy as np
import pandas as pd
import utils as ut
from config import (
    ED_FILENAME_REGIONS,
    ED_FILENAME_REGIONS_RANDOM,
    PATH_CLUSTERS,
    PATH_ED_REGIONS,
    REGIONS,
    VERSION,
)
from scipy import stats
from tqdm import tqdm

# %% LOAD DATA
df_clusters = pd.read_csv(
    os.path.join(PATH_CLUSTERS, VERSION, "df_clusters_full_ordered.csv"),
)
df_clusters["time"] = pd.to_datetime(df_clusters["time"])
df_clusters["time"] = df_clusters["time"].apply(
    lambda dt: dt.replace(hour=12, minute=0, second=0),
)  # set time to noon, to match energy data; is daily average anyway

ed = pd.read_csv(
    os.path.join(PATH_ED_REGIONS, ED_FILENAME_REGIONS),
).reset_index(drop=True)

ed["run"] = ed["runs"].str.extract(r"(\d+)").astype(int)
df_events = ed.drop(["Unnamed: 0", "runs"], axis=1)

# compare to dataframe of random 7-day windows, that are not necessarily energy droughts
random_ed = pd.read_csv(
    os.path.join(PATH_ED_REGIONS, ED_FILENAME_REGIONS_RANDOM),
).reset_index(drop=True)
random_ed["run"] = random_ed["runs"].str.extract(r"(\d+)").astype(int)
df_random = random_ed.drop(["Unnamed: 0", "runs"], axis=1)


# %%
# Defining functions
def calculate_persistency_stats(df, date_col="time", cluster_col="cluster_id"):
    df[date_col] = pd.to_datetime(df[date_col])

    df["date_diff"] = df.groupby(cluster_col)[date_col].diff().dt.days
    df["new_streak"] = df["date_diff"] != 1

    df["streak_id"] = df.groupby(cluster_col)["new_streak"].cumsum()

    streak_lengths = df.groupby([cluster_col, "streak_id"]).size().reset_index(name="streak_length")

    persistency_stats = streak_lengths.groupby(cluster_col).agg(["mean", "median", "std", "max"])

    return persistency_stats, streak_lengths


def calc_persistence_comparison(
    cluster_id,
    region,
    df_event_wr,
    df_random_wr,
    df,
    streak_lengths_Bayes,
):
    """Calculate the mean streak lengths for a given weather regime cluster,
    comparing event-specific means to the overall mean streak length for that cluster.

    Args:
    ----
    cluster_id (int): The ID of the cluster whose streak lengths are to be analyzed.
    df_event_wr (DataFrame): DataFrame containing event and weather regime information.
    df (DataFrame): DataFrame containing just weather regime data for matching and querying purposes.
    streak_lengths_Bayes (DataFrame): DataFrame containing streak lengths for Bayesian clusters.

    Returns:
    -------
    tuple: A tuple containing the mean streak length for the events and the streak length for the random events

    """
    streak_lengths_copy = streak_lengths_Bayes.copy()

    streak_length_array = ([], [])

    df_events_regime = df_event_wr.query(
        "country in @region and dominant_weather_regime == @cluster_id",
    )
    df_random_regime = df_random_wr.query(
        "country in @region and dominant_weather_regime == @cluster_id",
    )
    df_list = [df_events_regime, df_random_regime]
    err = [0, 0]
    for i, df_i in enumerate(df_list):
        for ind, row in df_i.iterrows():
            run = row["run"]
            start_time = pd.to_datetime(row["start_time"])

            first_time_index = np.where(np.array(row["weather_regime_ids"]) == cluster_id)[0][0]
            streak_time = start_time + pd.Timedelta(days=first_time_index)

            try:
                streak_id = df.query("run == @run and time == @streak_time")["streak_id"].values[0]
                streak_length = streak_lengths_copy.query(
                    "Bayes_cluster == @cluster_id and streak_id == @streak_id",
                )
                streak_length_value = streak_length["streak_length"].values[0]
                streak_length_array[i].append(streak_length_value)
            except IndexError:
                err[i] += 1
                continue
    return streak_length_array


def diff_persistence_error_calc(event_persistences, random_persistences):
    """Compares the persistence of weather regimes leading to events with random weather regimes,
    calculating mean difference, effect size (Cohen's d), and 95% confidence interval.

    Args:
    ----
    event_persistences (np.array): Array of persistence times for weather regimes leading to events.
    random_persistences (np.array): Array of persistence times for random weather regimes.

    Returns:
    -------
    dict: A dictionary containing the mean difference, p-value, and 95% confidence interval.

    """
    mean_diff = np.mean(event_persistences) - np.mean(random_persistences)
    perc_diff = (
        (np.mean(event_persistences) - np.mean(random_persistences)) / np.mean(random_persistences)
    ) * 100
    t_stat, p_value = stats.ttest_ind(event_persistences, random_persistences, equal_var=False)

    n1, n2 = len(event_persistences), len(random_persistences)
    s1, s2 = np.var(event_persistences, ddof=1), np.var(random_persistences, ddof=1)
    # not using this right now
    pooled_sd = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    cohen_d = mean_diff / pooled_sd

    # Calculate the standard error
    se = np.sqrt(s1 / n1 + s2 / n2)
    perc_se = 100 * se / np.mean(random_persistences)
    ci_low = mean_diff - 1.96 * se  # 95% CI
    ci_high = mean_diff + 1.96 * se  # 95% CI
    perc_ci_low = perc_diff - 1.96 * perc_se
    perc_ci_high = perc_diff + 1.96 * perc_se

    return {
        "Mean Difference": mean_diff,
        "Percentage Difference": perc_diff,
        "P-Value": p_value,
        "95% CI (Low)": ci_low,
        "95% CI (High)": ci_high,
        "95% CI Percentage (Low)": perc_ci_low,
        "95% CI Percentage (High)": perc_ci_high,
    }


def persistence_comparison(
    df_event_wr,
    df_random_wr,
    df_clusters,
    streak_lengths_Bayes,
    REGIONS,
):
    """Compares the persistence between events and random weather regimes (WRs) for each region and cluster.

    Args:
    ----
        df_event_wr: DataFrame containing event weather regimes.
        df_random_wr: DataFrame containing random weather regimes.
        df_clusters: DataFrame containing weather regime cluster information.
        streak_lengths_Bayes: Array or list of streak lengths computed using Bayesian methods.
        REGIONS: A dictionary mapping region keys to region names or identifiers.
        CLUSTER_NAMES: A list of names corresponding to each cluster for title annotation in plots.

    Returns:
    -------
        A dictionary containing the arrays of streak lengths for each region and cluster.

    """
    all_streak_arrays = {}

    for region_key in REGIONS.keys():
        # region = REGIONS[region_key]
        all_streak_arrays[region_key] = []

        # fig, axs = plt.subplots(2, 2)
        # axs = axs.flatten()
        # bins = np.arange(49, step=3)

        for i in tqdm(range(4)):
            streak_length_array = calc_persistence_comparison(
                i,
                region_key,
                df_event_wr,
                df_random_wr,
                df_clusters,
                streak_lengths_Bayes,
            )

            all_streak_arrays[region_key].append(streak_length_array)

    return all_streak_arrays


# %% ANALYSIS

# Calculate the persistence statistics for all Bayesian weather regimes, not related to events
pers_stats, streak_lengths = calculate_persistency_stats(
    df_clusters,
    cluster_col="Bayes_cluster",
)

df_event_wr = ut.find_dominant_wr(df_events, df_clusters, cluster_col="Bayes_cluster")
df_random_wr = ut.find_dominant_wr(df_random, df_clusters, cluster_col="Bayes_cluster")

# %%
streak_arrays = persistence_comparison(
    df_event_wr,
    df_random_wr,
    df_clusters,
    streak_lengths,
    REGIONS,
)

# %%
results_df = pd.DataFrame(
    columns=[
        "Region",
        "WR",
        "Mean Difference",
        "Percentage Difference",
        "P-Value",
        "95% CI (Low)",
        "95% CI (High)",
        "95% CI Percentage (Low)",
        "95% CI Percentage (High)",
    ],
)

result_rows = []

for region_key in REGIONS:
    for i in range(4):
        event_persistences = streak_arrays[region_key][i][0]
        random_persistences = streak_arrays[region_key][i][1]
        result = diff_persistence_error_calc(event_persistences, random_persistences)

        row = {"Region": region_key, "WR": i, **result}
        result_rows.append(row)

results_df = pd.concat([results_df, pd.DataFrame(result_rows)], ignore_index=True)
results_df.to_csv("../Results/data/Bayes_full_v2/df_persistence_diff.csv")
# %%
