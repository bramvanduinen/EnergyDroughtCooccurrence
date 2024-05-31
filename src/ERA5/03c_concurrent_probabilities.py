# %%
# IMPORTS
import os
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import utils as ut
from config_ERA5 import (
    COUNTRIES,
    PATH_CLUSTERS,
    PATH_DATA,
    PATH_ED,
    REGIONS,
    VERSION,
)
from matplotlib import cm, colors
from scipy.stats import norm

# %%
# LOAD DATA
HOMEDIR = "/usr/people/duinen/MSc-thesis/"
ROW_ORDER = np.load(
    f"{HOMEDIR}Data/row_order_nettodemand_v20240220.npy",
)  # load the row ordering of the clustered residual heatmap, to follow the same clustering!
CMAP = "RdBu_r"

BOUNDS_GREY = np.arange(-75, 85, 10)
CMAP_GREY = cm.get_cmap("RdBu_r", len(BOUNDS_GREY) - 1)
NORM_GREY = colors.BoundaryNorm(BOUNDS_GREY, CMAP_GREY.N)
CMAP_GREY.set_bad("grey")
# %%
ut.check_make_dir(os.path.join(PATH_DATA, VERSION))
DF_CO_OCCURRENCES_FILENAME = os.path.join(PATH_DATA, VERSION, "df_co_occurrences.csv")

df_clusters = pd.read_csv(
    os.path.join(PATH_CLUSTERS, VERSION, "df_clusters_full_ordered.csv"),
)
df_clusters["time"] = pd.to_datetime(df_clusters["time"])
df_clusters["time"] = df_clusters["time"].apply(
    lambda dt: dt.replace(hour=12, minute=0, second=0),
)  # set time to noon, to match df. Is daily average anyway

ed = pd.read_csv(
    os.path.join(
        PATH_ED,
        "demand_net_renewables_netto_demand_el7_winter_ERA5_2023_PD_noHydro_73_events_v2.csv",
    ),
).reset_index(drop=True)
df_events = ed.drop(["Unnamed: 0"], axis=1)

df_events.query("country.isin(@COUNTRIES)", inplace=True)
df_events["start_time"] = pd.to_datetime(df_events["start_time"]).apply(
    lambda dt: dt.replace(hour=12, minute=0, second=0),
)  # set time to noon, to match df. Is daily average anyway

df_events["end_time"] = pd.to_datetime(df_events["end_time"]).apply(
    lambda dt: dt.replace(hour=12, minute=0, second=0),
)  # set time to noon, to match df. Is daily average anyway


def find_dominant_cluster(window):
    counts = window.value_counts()
    max_count = counts.max()
    # Check if the maximum count is at least 4
    if max_count >= 4:
        return counts.idxmax()
    # Return 5 if no cluster is dominant
    return 5


df_clusters["dominant_cluster"] = (
    df_clusters["Bayes_cluster"]
    .rolling(window=7, min_periods=7)
    .apply(find_dominant_cluster, raw=False)
)
cluster_occurrences = plt.hist(df_clusters["dominant_cluster"], bins=np.arange(7), density=True)[0]

df_event_wr = df_events.merge(df_clusters, left_on="end_time", right_on="time")
# %%
# %%
# DEFINE FUNCTIONS


def get_co_occurrence(df_full, wr):
    overlapping_events_list = []
    co_occur_prob_list = []

    if wr != "all":
        df = df_full.query("dominant_weather_regime == @wr")
        num_events_for_wr = (
            df_events.query("dominant_weather_regime == @wr").groupby("country").size()
        )
    if wr == "all":
        num_events_for_wr = df_events.groupby("country").size()
        df = df_full.copy()

    country_combinations = product(df_full["country"].unique(), repeat=2)
    for country_1, country_2 in country_combinations:
        co_occur = 0  # reset counter
        events_country_1 = df_full.query("country == @country_1").sort_values(
            by="start_time",
        )
        events_country_2 = df_full.query("country == @country_2").sort_values(
            by="start_time",
        )

        for _, event_country_1 in events_country_1.iterrows():
            for _, event_country_2 in events_country_2.iterrows():
                if (
                    event_country_1["end_time"] >= event_country_2["start_time"]
                    and event_country_2["end_time"] >= event_country_1["start_time"]
                ):
                    co_occur += 1

                    overlapping_events_list.append(
                        {
                            "country_1": country_1,
                            "event_number_1": event_country_1["event_number"],
                            "start_time_1": event_country_1["start_time"],
                            "end_time_1": event_country_1["end_time"],
                            "country_2": country_2,
                            "event_number_2": event_country_2["event_number"],
                            "start_time_2": event_country_2["start_time"],
                            "end_time_2": event_country_2["end_time"],
                        },
                    )

        co_occur_prob_list.append(
            {
                "country_1": country_1,
                "country_2": country_2,
                "count": co_occur,
            },
        )

    overlapping_events_df = pd.DataFrame(overlapping_events_list)
    co_occur_prob_df = pd.DataFrame(co_occur_prob_list)

    co_occur_count_df = co_occur_prob_df.groupby(["country_1", "country_2"], as_index=False)[
        "count"
    ].sum()

    return overlapping_events_df, co_occur_prob_list, co_occur_count_df


def count_to_conditional(co_occur_count_df, country_order):
    pivot_table = (
        co_occur_count_df.pivot(index="country_1", columns="country_2", values="count")
        .reindex(country_order)
        .reindex(columns=country_order)
    )

    base_rates = np.diag(pivot_table).copy()

    conditional = pivot_table / base_rates[:, np.newaxis]
    return conditional, base_rates


def p_value(meanprob, std, meanprob_all, std_all):
    SE = np.sqrt(std**2 + std_all**2)
    diff_means = meanprob - meanprob_all
    Z = diff_means / SE
    p_value = 2 * (1 - norm.cdf(abs(Z)))
    return p_value


# TODO: probably move these both to utils
def get_color(country):
    """Returns the color associated with a given country based on its regional classification dictionary."""
    for region in REGIONS.values():
        if country in region["countries"]:
            return region["color"]
    return None


def reorder(matrix):
    df_matrix = pd.DataFrame(matrix)

    matrix_reordered = df_matrix.reindex(ROW_ORDER, axis=0)
    matrix_reordered = matrix_reordered.reindex(ROW_ORDER, axis=1)
    return matrix_reordered.values


# %% CO-OCCURRENCE ANALYSIS

if not os.path.exists(DF_CO_OCCURRENCES_FILENAME):
    print("Computing co-occurrences...")
    events_df, counts_per_run, co_occur_counts_df = get_co_occurrence(df_full=df_events, wr="all")
    co_occur_counts_df.to_csv(DF_CO_OCCURRENCES_FILENAME, index=False)
    # counts_per_run.to_csv("counts_per_run.csv")
else:
    print("Loading previously computed co-occurrences...")
    co_occur_counts_df = pd.read_csv(DF_CO_OCCURRENCES_FILENAME)
    # counts_per_run = pd.read_csv("counts_per_run.csv")

# %%
# PLOT HEATMAP ALL WEATHER REGIMES, LINKAGE = DEFAULT AND COMPLETE
co_occur_counts_df["country_1_index"] = co_occur_counts_df["country_1"]
co_occur_counts_df["country_2_index"] = co_occur_counts_df["country_2"]

co_occur_counts_df["prob"] = co_occur_counts_df["count"] / 73
co_occur_counts_df["prob"] = pd.to_numeric(co_occur_counts_df["prob"])

heatmap_data = co_occur_counts_df.pivot(
    index=["country_1_index"],
    columns=["country_2_index"],
    values="prob",
)

bounds = np.arange(0, 1.1, 0.1)

# Create a colormap with the specified boundaries
cmap = colors.ListedColormap(sns.color_palette("Reds", len(bounds) - 1))
norm_cmap = colors.BoundaryNorm(bounds, cmap.N)

plt.figure(figsize=(11, 8), dpi=300)
cg = sns.clustermap(
    heatmap_data.values,
    cmap=cmap,
    norm=norm_cmap,
    xticklabels=COUNTRIES,
    yticklabels=COUNTRIES,
)
cg.ax_heatmap.set_xticklabels(cg.ax_heatmap.get_xmajorticklabels(), fontsize=16)
cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_ymajorticklabels(), fontsize=16)


cg.ax_col_dendrogram.set_visible(False)
x0, _y0, _w, _h = cg.cbar_pos
cg.ax_cbar.set_position([1.02, 0.06, 0.025, 0.74])
cg.cax.set_ylabel(r"P($E_{c,column}$|$E_{c,row}$)", size=18, rotation=90, labelpad=20)
cg.cax.yaxis.set_label_coords(x=3.5, y=0.5)
cg.cax.tick_params(labelsize=14)
cg.cax.yaxis.set_ticks(bounds)
row_order = cg.dendrogram_row.reordered_ind

# %%
# plot events over time, to see if de-trending is necessary
region_names = list(REGIONS.keys())
plt.figure(figsize=(10, 15))

for i, region in enumerate(region_names):
    df_events_r = df_events[df_events["country"].isin(REGIONS[region])]
    years_r = df_events_r["start_time"].dt.year

    plt.subplot(3, 2, i + 1)

    years_r.hist(bins=20)
    total = np.sum(years_r < 2100)
    first_half = 100 * np.sum(years_r < 1987) / total
    second_half = 100 - first_half
    plt.title(f"Region = {region}, First half: {first_half:.2f}%, second half: {second_half:.2f}%")
plt.tight_layout()
# %%
co_occurrence_all = reorder(heatmap_data.values)

plt.figure(figsize=(11, 8), dpi=300)
cg = sns.heatmap(
    co_occurrence_all,
    cmap=cmap,
    norm=norm_cmap,
    xticklabels=np.array(COUNTRIES)[ROW_ORDER],
    yticklabels=np.array(COUNTRIES)[ROW_ORDER],
    cbar=False,
)
cg.set_xticklabels(cg.get_xmajorticklabels(), fontsize=16, rotation=90)
cg.set_yticklabels(cg.get_ymajorticklabels(), fontsize=16)
cbar = plt.colorbar(cg.collections[0])
cbar.set_label(r"P($E_{c,column}$|$E_{c,row}$)", size=18)
cbar.set_ticks(bounds)
