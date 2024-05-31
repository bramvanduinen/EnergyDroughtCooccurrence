# %%
# IMPORTS
import os
from datetime import datetime

import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plot_utils as put
import utils as ut
from config import CLUSTER_NAMES, ED_FILENAME_REGIONS, PATH_CLUSTERS, PATH_ED, REGIONS, VERSION
from matplotlib import colors

# %%
REGION_LIST = list(REGIONS.keys())

# ANOM_PATH = "/net/pc230050/nobackup/users/duinen/LENTIS/present/"
# PSL_PATH = "/net/pc200256/nobackup/users/most/LENTIS/present/day/psl_d/"
# WINDOW = 7
# VARIABLE = "residual"  # option 2: total_RE
# CLUSTER_REMAPPING = {
#     0: 2,
#     1: 0,
#     2: 1,
#     3: 3,
#     4: 4,
# }  # make sure to change after new clustering run! remap the clusters to the correct order; so that names are as below,
# CLUSTER_NAMES = {0: "NAO -", 1: "NAO +", 2: "Blocking", 3: "Atl. Ridge"}
drop_no_regime = (
    True  # set to true if  you want to drop events that occur under 'No-regime' weather regime
)

date_now = datetime.now()
rundate = f"{date_now.year:04}{date_now.month:02}{date_now.day:02}"

df_clusters = pd.read_csv(
    os.path.join(PATH_CLUSTERS, VERSION, "df_clusters_full_ordered.csv"),
)
df_clusters["time"] = pd.to_datetime(df_clusters["time"])
df_clusters["time"] = df_clusters["time"].apply(
    lambda dt: dt.replace(hour=12, minute=0, second=0),
)  # set time to noon, to match df. Is daily average anyway

df_clusters["dominant_cluster"] = (
    df_clusters["Bayes_cluster"]
    .rolling(window=7, min_periods=7)
    .apply(ut.find_dominant_wr_v2(window=7), raw=False)
)

real_wrs = df_clusters["dominant_cluster"] <= 3
cluster_occurrence_rate = df_clusters.loc[real_wrs, "dominant_cluster"].value_counts(
    normalize=True,
)

# %%
ed = pd.read_csv(
    os.path.join(PATH_ED, ED_FILENAME_REGIONS),
).reset_index(drop=True)
ed["run"] = ed["runs"].str.extract(r"(\d+)").astype(int)
df_events = ed.drop(["Unnamed: 0", "runs"], axis=1)


# %%
def cluster_risk(df, region):
    """Calculate the risk factor (w.r.t. climatology) of energy drought events within each weather regime for a given list of countries.

    Parameters
    ----------
        df (pd.DataFrame): DataFrame containing the data.
        country_list (list): List of country names to filter the data.

    Returns
    -------
        list: List of relative risks (climatology = 1) for each weather regime cluster.

    """
    df_all = df.query("country == @region and 0 <= dominant_weather_regime <= 3")
    risks = []

    for i in range(4):
        df_ci = df.query("country == @region and dominant_weather_regime == @i")
        r_ci = (len(df_ci) / len(df_all) * 100) / (100 * cluster_occurrence_rate[i])
        risks.append(r_ci)

    # df_c0 = df.query("country in @country_list and dominant_weather_regime == 0")
    # df_c1 = df.query("country in @country_list and dominant_weather_regime == 1")
    # df_c2 = df.query("country in @country_list and dominant_weather_regime == 2")
    # df_c3 = df.query("country in @country_list and dominant_weather_regime == 3")
    # df_all = df.query("country in @country_list and 0 <= dominant_weather_regime <= 3")

    # # TODO: this should be the percentage of occurrence of the WR!!
    # r_c0 = (len(df_c0) / len(df_all) * 100) / cluster_occurrence_rate[0]
    # r_c1 = (len(df_c1) / len(df_all) * 100) / cluster_occurrence_rate[0]
    # r_c2 = (len(df_c2) / len(df_all) * 100) / cluster_occurrence_rate[0]
    # r_c3 = (len(df_c3) / len(df_all) * 100) / cluster_occurrence_rate[0]
    # risks = [r_c0, r_c1, r_c2, r_c3]
    # return np.array(risks)
    return np.array(risks)


# %%
def plot_weather_regimes(
    df,
    country_name,
    dtype,
    anom_data,
    data_psl,
    lons,
    lats,
    cluster_names,
    dataproj=ccrs.PlateCarree(),
    cmap=plt.cm.RdBu_r,
):
    """Plot weather regime-specific anomalies for a given country and drought intensity, save as image file.

    Parameters
    ----------
        df (pd.DataFrame): DataFrame containing the event data.
        country_name (str): Name of the country to analyze.
        dtype (str): Type of meteorological variable ('tas', 'sfcWind', 'rsds').
        anom_data (xr.Dataset): Anomalies data.
        data_psl (xr.Dataset): Surface pressure data.
        lons (np.ndarray): Longitudes array.
        lats (np.ndarray): Latitudes array.
        cluster_names (list): List of weather regime cluster names.
        dataproj (cartopy.crs.Projection, optional): Map projection. Defaults to PlateCarree.
        cmap (matplotlib.colors.Colormap, optional): Colormap for contour plot. Defaults to RdBu_r.

    Returns
    -------
        None

    """
    df_country = df.query("country == @country_name")

    vmin_vmax_dict = {
        "mild": {"tas": (-4, 4), "sfcWind": (-4, 4), "rsds": (-10, 10)},
        "moderate": {"tas": (-6, 6), "sfcWind": (-5, 5), "rsds": (-20, 20)},
        "severe": {"tas": (-8, 8), "sfcWind": (-6, 6), "rsds": (-25, 25)},
        "extreme": {"tas": (-10, 10), "sfcWind": (-7, 7), "rsds": (-30, 30)},
        "all": {"tas": (-3, 3), "sfcWind": (-3, 3), "rsds": (-10, 10)},
    }

    vmin, vmax = vmin_vmax_dict["extreme"][dtype]  # HACK: hardcode

    plot_titles = {
        "tas": ("Temperature", "[K]"),
        "sfcWind": ("Wind speed", "[m/s]"),
        "rsds": ("Solar irradiation", "[W/m2]"),
    }

    title_name, cbar_unit = plot_titles[dtype]

    fig, axs = plt.subplots(2, 2, figsize=(16, 8), subplot_kw={"projection": dataproj})
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])

    levels = np.linspace(vmin, vmax, 11)
    lon_lim, lat_lim = 40, 75
    for cluster_id in range(4):
        df_plot = df_country.query("dominant_weather_regime == @cluster_id")
        anom_plot = ut.calc_composite_mean_multipledayevent(anom_data, df_plot)
        psl_plot = ut.calc_composite_mean_multipledayevent(data_psl, df_plot)

        psl_plot = psl_plot / 100  # Pa to hPa
        ax = axs.flat[cluster_id]
        ax.set_extent([lons[0], lon_lim, lats[0], lat_lim])
        ax.set_ylim([lats[0], 75])

        norm = colors.BoundaryNorm(levels, ncolors=cmap.N, extend="both")
        im = ax.contourf(lons, lats, anom_plot, levels=levels, cmap=cmap, norm=norm, extend="both")
        CS = ax.contour(lons, lats, psl_plot, colors="k")
        ax.clabel(CS, inline=True, fontsize=10)
        put.plot_maxmin_points(
            ax,
            lons,
            lats,
            lon_lim,
            lat_lim,
            psl_plot,
            "max",
            50,
            "H",
            color="k",
            plotValue=False,
            transform=dataproj,
        )
        put.plot_maxmin_points(
            ax,
            lons,
            lats,
            lon_lim,
            lat_lim,
            psl_plot,
            "min",
            50,
            "L",
            color="k",
            plotValue=False,
            transform=dataproj,
        )
        ax.coastlines()
        ax.add_feature(cartopy.feature.BORDERS, linestyle=":", alpha=1)
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        ax.set_title(f"{cluster_names[cluster_id]}, n = {len(df_plot)}")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig.colorbar(sm, cax=cbar_ax, label=cbar_unit)
    fig.suptitle(f"{title_name} anomalies per WR events in {country_name}", fontsize=16)

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    # plt.savefig(
    #     f"{dir_MeteoFigures}{country_name}_{dtype}_anomalies.png",
    #     dpi=300,
    #     bbox_inches="tight",
    # )


# %%
df_event_wr = ut.find_dominant_wr(df_events, df_clusters, cluster_col="Bayes_cluster")

# %% FIND RISK RATIO PER WR


risk_ratios = []
for region in REGION_LIST:
    risk_ratios.append(cluster_risk(df_event_wr, region) - 1)

# %%
labels = [CLUSTER_NAMES[0], CLUSTER_NAMES[1], CLUSTER_NAMES[2], CLUSTER_NAMES[3]]

num_conditions = len(labels)

width = 0.15
positions = np.arange(num_conditions)

region_colors = {
    "Nordic": "C0",
    "Baltic": "cyan",
    "Northwestern": "C1",
    "Central": "C2",
    "Eastern": "C3",
    "Iberia": "C4",
}

plt.figure(dpi=300, figsize=(8, 6))
plt.bar(
    positions - 3 * width,
    risk_ratios[2],
    width,
    label="Nordic",
    color=region_colors["Nordic"],
    edgecolor="black",
    bottom=1,
)
plt.bar(
    positions - 2 * width,
    risk_ratios[3],
    width,
    label="Baltic",
    color=region_colors["Baltic"],
    edgecolor="black",
    bottom=1,
)
plt.bar(
    positions - width,
    risk_ratios[1],
    width,
    label="Northwestern",
    color=region_colors["Northwestern"],
    edgecolor="black",
    bottom=1,
)
plt.bar(
    positions,
    risk_ratios[4],
    width,
    label="Central",
    color=region_colors["Central"],
    edgecolor="black",
    bottom=1,
)
plt.bar(
    positions + width,
    risk_ratios[5],
    width,
    label="Eastern",
    color=region_colors["Eastern"],
    edgecolor="black",
    bottom=1,
)
plt.bar(
    positions + 2 * width,
    risk_ratios[0],
    width,
    label="Iberia",
    color=region_colors["Iberia"],
    edgecolor="black",
    bottom=1,
)

# Get the properties of the y-axis line
ax = plt.gca()
yaxis_line = ax.get_yaxis().get_gridlines()[0]
line_width = yaxis_line.get_linewidth()

# Draw the axhline with the same color and linewidth as the y-axis line
plt.axhline(y=1, color="k", linestyle="-", linewidth=line_width)

plt.ylabel("Risk ratio", fontsize=14)

plt.yscale("log")
plt.yticks([0.2, 0.5, 1, 2, 5], [0.2, 0.5, 1, 2, 5], fontsize=14)
plt.xticks(positions, labels, fontsize=14)
plt.legend(fontsize=12)
# plt.savefig("../Results/Figures/risk_ratios.png", dpi=300, bbox_inches="tight")

# %%
