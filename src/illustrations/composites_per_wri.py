# %%
import ast
import os

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils as ut
import xarray as xr
from config import CLUSTER_NAMES, PATH_ANOM, PATH_CLUSTERS, PATH_PSL, VERSION
from matplotlib import colors
from tqdm.notebook import tqdm

# %%

# LOAD DATA

df_clusters = pd.read_csv(
    os.path.join(PATH_CLUSTERS, VERSION, "df_clusters_full_ordered.csv"),
)
df_clusters["time"] = pd.to_datetime(df_clusters["time"])
df_clusters["time"] = df_clusters["time"].apply(
    lambda dt: dt.replace(hour=12, minute=0, second=0),
)  # set time to noon, to match df. Is daily average anyway

dir_Figures = "/usr/people/duinen/MSc-thesis/Results/Figures/composites_wri"
dir_Data = "/net/pc230050/nobackup/users/duinen/results/composites_wri"

ut.check_make_dir(dir_Figures)
ut.check_make_dir(dir_Data)

# dataproj = ccrs.LambertConformal(central_longitude=0, central_latitude=45)
dataproj = ccrs.PlateCarree()
cmap = plt.cm.RdBu_r

# changed yellow to orange, for better visibility
GyYl = colors.LinearSegmentedColormap.from_list(
    "my_colormap",
    [(0, "grey"), (0.5, "white"), (1, "orange")],  # (position, color)
)

BrBl = colors.LinearSegmentedColormap.from_list(
    "my_colormap",
    [(0, "brown"), (0.5, "white"), (1, "blue")],  # (position, color)
)
# %% Add WRI at assigned cluster
all_wri = np.array(df_clusters["wri"].apply(ast.literal_eval).to_list())
assigned_clusters = np.array(df_clusters["Bayes_cluster_raw"])
wri_at_cluster = np.array(all_wri)[np.arange(len(assigned_clusters)), assigned_clusters]
df_clusters["wri_at_cluster"] = wri_at_cluster


# %%
# def create_composite_dataset(df_clusters, wr, quantile):
#     thres = np.quantile(df_clusters.query("Bayes_cluster == @wr")["wri_at_cluster"], quantile)

#     # composite_dataset = []
#     for run in tqdm(np.arange(10, 170)):
#         times_run = np.array(
#             df_clusters.query(
#                 "run == @run and Bayes_cluster == @wr and wri_at_cluster > @thres",
#             )["time"],
#         )

#         rsds_run = xr.open_dataset(
#             f"{PATH_ANOM}/rsds_d_anomaly/anom_rsds_d_ECEarth3_h{run:03d}.nc",
#         )["rsds"]
#         sfcWind_run = xr.open_dataset(
#             f"{PATH_ANOM}/sfcWind_d_anomaly/anom_sfcWind_d_ECEarth3_h{run:03d}.nc",
#         )["sfcWind"].drop_vars("height")
#         t2m_run = xr.open_dataset(
#             f"{PATH_ANOM}/tas_d_anomaly/anom_tas_d_ECEarth3_h{run:03d}.nc",
#         )["tas"].drop_vars("height")
#         pr_run = xr.open_dataset(
#             f"{PATH_ANOM}/pr_d_anomaly/anom_pr_d_ECEarth3_h{run:03d}.nc",
#         )["pr"]
#         psl_run = xr.open_dataset(f"{PATH_PSL}/psl_d_ECEarth3_h{run:03d}.nc")["psl"]

#         rsds_subset = rsds_run.sel(time=times_run)
#         sfcWind_subset = sfcWind_run.sel(time=times_run)
#         t2m_subset = t2m_run.sel(time=times_run)
#         pr_subset = pr_run.sel(time=times_run)
#         psl_subset = psl_run.sel(time=times_run)

#         temp_dataset = xr.Dataset(
#             {
#                 "rsds": rsds_subset,
#                 "sfcWind": sfcWind_subset,
#                 "tas": t2m_subset,
#                 "pr": pr_subset,
#                 "psl": psl_subset,
#             },
#             coords={
#                 "time": times_run,
#                 "lon": rsds_subset.lon,
#                 "lat": rsds_subset.lat,
#                 "run": run,
#             },
#         )
#         if run == 10:
#             composite_dataset = temp_dataset
#         else:
#             composite_dataset = xr.concat([composite_dataset, temp_dataset], dim="run")
#             temp_dataset.close()

#     return composite_dataset


def create_weighted_composite_means(df_clusters, wr, quantile, path_anom, path_psl):
    """Computes the weighted mean of specified datasets based on cluster information, quantile threshold,
    and the number of time entries per run.

    Args:
    ----
        df_clusters (DataFrame): DataFrame containing cluster data.
        wr (int): Bayes cluster identifier to filter data.
        quantile (float): Quantile value to determine threshold within clusters.
        path_anom (str): Path to anomaly data files.
        path_psl (str): Path to psl data files.

    Returns:
    -------
        xarray.Dataset: Dataset containing the weighted means for rsds, sfcWind, tas, and psl.

    """
    thres_min = np.quantile(df_clusters.query("Bayes_cluster == @wr")["wri_at_cluster"], quantile)
    thres_max = np.quantile(
        df_clusters.query("Bayes_cluster == @wr")["wri_at_cluster"],
        quantile + 0.1,
    )

    sums = {}
    counts = {}

    for run in tqdm(range(10, 170)):
        # composite of wri range
        times_run = df_clusters.query(
            "run == @run and Bayes_cluster == @wr and wri_at_cluster >= @thres_min and wri_at_cluster < @thres_max",
        )["time"].to_numpy()
        # composite of all!
        # times_run = df_clusters.query(
        #     "run == @run and Bayes_cluster == @wr",
        # )["time"].to_numpy()

        if len(times_run) == 0:
            continue

        rsds = xr.open_dataset(f"{path_anom}/rsds_d_anomaly/anom_rsds_d_ECEarth3_h{run:03d}.nc")[
            "rsds"
        ]
        sfcWind = xr.open_dataset(
            f"{path_anom}/sfcWind_d_anomaly/anom_sfcWind_d_ECEarth3_h{run:03d}.nc",
        )["sfcWind"].drop_vars("height")
        tas = xr.open_dataset(f"{path_anom}/tas_d_anomaly/anom_tas_d_ECEarth3_h{run:03d}.nc")[
            "tas"
        ].drop_vars("height")
        pr = xr.open_dataset(f"{path_anom}/pr_d_anomaly/anom_pr_d_ECEarth3_h{run:03d}.nc")["pr"]
        psl = xr.open_dataset(f"{path_psl}/psl_d_ECEarth3_h{run:03d}.nc")["psl"]

        for var_name, ds in [
            ("rsds", rsds),
            ("sfcWind", sfcWind),
            ("tas", tas),
            ("pr", pr),
            ("psl", psl),
        ]:
            selected = ds.sel(time=times_run)
            sum_data = selected.sum(dim="time", skipna=True)
            # count_data = len(times_run)
            count_data = selected.count(dim="time")

            if var_name not in sums:
                sums[var_name] = sum_data
                counts[var_name] = count_data
            else:
                sums[var_name] = sums[var_name] + sum_data
                counts[var_name] = counts[var_name] + count_data

        rsds.close()
        sfcWind.close()
        tas.close()
        pr.close()
        psl.close()

    weighted_means = {var: sums[var] / counts[var] for var in sums}
    # HACK: for some reason, precipitation data does not go well in the full dataset, so separate.
    return xr.Dataset(weighted_means), weighted_means["pr"]


# %%

composite_wr = {}
composite_wr_pr = {}
quantile = 0.90
for i, regime in tqdm(enumerate(CLUSTER_NAMES)):
    composite_wr1, composite_pr_wr1 = create_weighted_composite_means(
        df_clusters,
        i,
        quantile,
        PATH_ANOM,
        PATH_PSL,
    )
    composite_wr[regime] = composite_wr1
    composite_wr_pr[regime] = composite_pr_wr1


# %%
fontsize = 16


def plot_composite_meteorology(composite_dataset, composite_dataset_pr, quantile):
    """Plot composite meteorological data for different weather regimes.

    Parameters
    ----------
    composite_dataset (xr.Dataset): Composite dataset with meteorological variables.
    wr (str): Weather regime name.
    quantile (float): Quantile value for plotting.
    dir_figures (str): Directory path to save the figures.

    """
    lons = composite_dataset["Blocking"].lon.values
    lats = composite_dataset["Blocking"].lat.values
    lons_plot = lons[(lons < 33) & (lons > -30)]
    lats_plot = lats[(lats < 70) & (lats > 35)]

    fig, axs = plt.subplots(
        4,
        4,
        figsize=(8.27, 15),
        dpi=300,
        subplot_kw={"projection": ccrs.EuroPP()},
    )
    fig.subplots_adjust(left=0.02, bottom=0.05, right=0.98, top=0.95, wspace=0.08, hspace=0.2)

    # vmin_list = [-5, -5, -30]
    # vmax_list = [5, 5, 30]
    levels_lists = [
        [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
        [-3, -2.5, -2, -1.5, -1.0, 1.0, 1.5, 2.0, 2.5, 3.0],
        [-25, -20, -15, -10, -5, 5, 10, 15, 20, 25],
        [-3, -2.5, -2, -1.5, -1.0, 1.0, 1.5, 2.0, 2.5, 3.0],
    ]

    titles = ["2 m temperature", "10 m wind speed", "Solar radiation", "Precipitation"]
    alphabet = [
        "(a)",
        "(b)",
        "(c)",
        "(d)",
        "(e)",
        "(f)",
        "(g)",
        "(h)",
        "(i)",
        "(j)",
        "(k)",
        "(l)",
        "(m)",
        "(n)",
        "(o)",
        "(p)",
    ]
    cbar_units = ["[K]", "[m/s]", "[W/m²]", "[mm/day]"]
    cmaps = [plt.cm.RdBu_r, plt.cm.PiYG, GyYl, plt.cm.BrBG]

    for j, wr in enumerate(CLUSTER_NAMES):
        composite_dataset_wr = composite_dataset[wr]
        composite_dataset_pr_wr = composite_dataset_pr[wr]
        data = [
            composite_dataset_wr.tas,
            composite_dataset_wr.sfcWind,
            composite_dataset_wr.rsds,
            composite_dataset_pr_wr * 86400,  # convert kgm2/s to mm/day
        ]

        # fig.text(0.22 + j * 0.20, 0.97, wr, ha="center", va="top", fontsize=16)
        if j == 1:
            wr = "NAO \u2212"

        for i in range(4):
            anom_plot = data[i]
            levels = levels_lists[i]
            ax = axs[i, j]
            if i == 0:
                ax.set_title(wr, pad=25, size=fontsize)
            if j == 0:
                ax.text(
                    -0.1,
                    0.55,
                    titles[i],
                    va="bottom",
                    ha="center",
                    rotation="vertical",
                    rotation_mode="anchor",
                    transform=ax.transAxes,
                    fontsize=fontsize - 2,
                )

            psl_plot = composite_dataset_wr.psl / 100  # Pa to hPa
            psl_plothl = psl_plot.where(
                (psl_plot.lon > -30)
                & (psl_plot.lon < 33)
                & (psl_plot.lat > 35)
                & (psl_plot.lat < 70),
                drop=True,
            )
            ax.set_extent([-30, 33, 35, 70], crs=ccrs.PlateCarree())

            norm = colors.BoundaryNorm(levels, ncolors=plt.cm.get_cmap(cmaps[i]).N, extend="both")
            if i != 3:
                im = ax.contourf(
                    lons,
                    lats,
                    anom_plot,
                    transform=ccrs.PlateCarree(),
                    levels=levels,
                    cmap=cmaps[i],
                    norm=norm,
                    extend="both",
                )
            elif i == 3:
                im = ax.contourf(
                    anom_plot.lon.values,
                    anom_plot.lat.values,
                    anom_plot,
                    transform=ccrs.PlateCarree(),
                    levels=levels,
                    cmap=cmaps[i],
                    norm=norm,
                    extend="both",
                )

            if j == 3:
                cbar_ax = fig.add_axes([0.2, 0.04 + (3 - i) * 0.235, 0.6, 0.01])
                cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
                cbar.set_label(cbar_units[i], fontsize=fontsize - 2)
                cbar.ax.tick_params(labelsize=fontsize - 2)

            CS = ax.contour(
                lons,
                lats,
                psl_plot,
                transform=ccrs.PlateCarree(),
                colors="0.4",
                linewidth=0.2,
            )
            ax.clabel(CS, inline=True, fontsize=10)
            # put.plot_maxmin_points(
            #     ax,
            #     lons_plot,
            #     lats_plot,
            #     33,
            #     70,
            #     psl_plothl,
            #     "max",
            #     50,
            #     "H",
            #     color="k",
            #     plotValue=False,
            #     transform=ccrs.PlateCarree(),
            # )
            # put.plot_maxmin_points(
            #     ax,
            #     lons_plot,
            #     lats_plot,
            #     33,
            #     70,
            #     psl_plothl,
            #     "min",
            #     50,
            #     "L",
            #     color="k",
            #     plotValue=False,
            #     transform=ccrs.PlateCarree(),
            # )
            ax.coastlines()
            ax.set_title(alphabet[i * 4 + j], loc="left", pad=5, size=14)
            # if j == 0:
            #     ax.set_title(alphabet[i * 4] + "  " + titles[i], loc="left", pad=5, size=14)
            # else:
            #     ax.set_title(alphabet[i * 4 + j], loc="left", pad=5, size=14)

    # plt.tight_layout()
    plt.show()


plot_composite_meteorology(composite_wr, composite_wr_pr, quantile)

# %%
