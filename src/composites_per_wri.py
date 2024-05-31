# %%
import ast
import os

import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plot_utils as put
import utils as ut
import xarray as xr
from config import CLUSTER_NAMES, PATH_ANOM, PATH_CLUSTERS, PATH_PSL, VERSION
from matplotlib import colors, ticker
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

dataproj = ccrs.LambertConformal(central_longitude=0, central_latitude=45)
cmap = plt.cm.RdBu_r

GyYl = colors.LinearSegmentedColormap.from_list(
    "my_colormap",
    [(0, "grey"), (0.5, "white"), (1, "yellow")],  # (position, color)
)
# %% Add WRI at assigned cluster
all_wri = np.array(df_clusters["wri"].apply(ast.literal_eval).to_list())
assigned_clusters = np.array(df_clusters["Bayes_cluster_raw"])
wri_at_cluster = np.array(all_wri)[np.arange(len(assigned_clusters)), assigned_clusters]
df_clusters["wri_at_cluster"] = wri_at_cluster


# %%
def create_composite_dataset(df_clusters, wr, quantile):
    thres = np.quantile(df_clusters.query("Bayes_cluster == @wr")["wri_at_cluster"], quantile)

    # composite_dataset = []
    for run in tqdm(np.arange(10, 170)):
        times_run = np.array(
            df_clusters.query(
                "run == @run and Bayes_cluster == @wr and wri_at_cluster > @thres",
            )["time"],
        )

        rsds_run = xr.open_dataset(
            f"{PATH_ANOM}/rsds_d_anomaly/anom_rsds_d_ECEarth3_h{run:03d}.nc",
        )["rsds"]
        sfcWind_run = xr.open_dataset(
            f"{PATH_ANOM}/sfcWind_d_anomaly/anom_sfcWind_d_ECEarth3_h{run:03d}.nc",
        )["sfcWind"].drop_vars("height")
        t2m_run = xr.open_dataset(
            f"{PATH_ANOM}/tas_d_anomaly/anom_tas_d_ECEarth3_h{run:03d}.nc",
        )["tas"].drop_vars("height")
        psl_run = xr.open_dataset(f"{PATH_PSL}/psl_d_ECEarth3_h{run:03d}.nc")["psl"]

        rsds_subset = rsds_run.sel(time=times_run)
        sfcWind_subset = sfcWind_run.sel(time=times_run)
        t2m_subset = t2m_run.sel(time=times_run)
        psl_subset = psl_run.sel(time=times_run)

        temp_dataset = xr.Dataset(
            {
                "rsds": rsds_subset,
                "sfcWind": sfcWind_subset,
                "tas": t2m_subset,
                "psl": psl_subset,
            },
            coords={
                "time": times_run,
                "lon": rsds_subset.lon,
                "lat": rsds_subset.lat,
                "run": run,
            },
        )
        if run == 10:
            composite_dataset = temp_dataset
        else:
            composite_dataset = xr.concat([composite_dataset, temp_dataset], dim="run")
            temp_dataset.close()

    return composite_dataset


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
        quantile + 0.05,
    )
    sums = {}
    counts = {}

    for run in tqdm(range(10, 170)):
        times_run = df_clusters.query(
            "run == @run and Bayes_cluster == @wr and wri_at_cluster >= @thres_min and wri_at_cluster < @thres_max",
        )["time"].to_numpy()

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
        psl = xr.open_dataset(f"{path_psl}/psl_d_ECEarth3_h{run:03d}.nc")["psl"]

        for var_name, ds in [("rsds", rsds), ("sfcWind", sfcWind), ("tas", tas), ("psl", psl)]:
            selected = ds.sel(time=times_run)
            sum_data = selected.sum(dim="time")
            count_data = len(times_run)

            if var_name not in sums:
                sums[var_name] = sum_data
                counts[var_name] = count_data
            else:
                sums[var_name] += sum_data
                counts[var_name] += count_data

        rsds.close()
        sfcWind.close()
        tas.close()
        psl.close()

    weighted_means = {var: sums[var] / counts[var] for var in sums}

    return xr.Dataset(weighted_means)


def plot_composite_meteorology(composite_dataset, wr, quantile):
    lons = composite_dataset.lon.values
    lats = composite_dataset.lat.values
    lons_plot = lons[(lons < 33) & (lons > -30)]
    lats_plot = lats[(lats < 70) & (lats > 35)]

    fig, axs = plt.subplots(3, 1, figsize=(6, 10), dpi=300, subplot_kw={"projection": dataproj})
    fig.subplots_adjust(right=0.8)

    vmin_list = [-5, -5, -30]
    vmax_list = [5, 5, 30]

    data = [composite_dataset.tas, composite_dataset.sfcWind, composite_dataset.rsds]
    titles = ["Temperature", "Wind speed", "Radiation"]
    cbar_units = ["[K]", "[m/s]", "[W/m²]"]
    cmaps = [plt.cm.RdBu_r, plt.cm.PiYG, GyYl]
    lon_lim, lat_lim = 33, 70

    for i in range(3):
        anom_plot = data[i]
        levels = np.linspace(vmin_list[i], vmax_list[i], 11)
        ax = axs.flat[i]
        psl_plot = composite_dataset.psl / 100  # Pa to hPa
        psl_plothl = psl_plot.where(
            (psl_plot.lon > -30) & (psl_plot.lon < 33) & (psl_plot.lat > 35) & (psl_plot.lat < 70),
            drop=True,
        )
        ax.set_extent([-30, 33, 35, 70], crs=ccrs.PlateCarree())

        norm = colors.BoundaryNorm(levels, ncolors=cmap.N, extend="both")
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
        fig.colorbar(im, ax=ax, label=cbar_units[i])
        CS = ax.contour(lons, lats, psl_plot, transform=ccrs.PlateCarree(), colors="k")
        ax.clabel(CS, inline=True, fontsize=10)
        put.plot_maxmin_points(
            ax,
            lons_plot,
            lats_plot,
            lon_lim,
            lat_lim,
            psl_plothl,
            "max",
            50,
            "H",
            color="k",
            plotValue=False,
            transform=ccrs.PlateCarree(),
        )
        put.plot_maxmin_points(
            ax,
            lons_plot,
            lats_plot,
            lon_lim,
            lat_lim,
            psl_plothl,
            "min",
            50,
            "L",
            color="k",
            plotValue=False,
            transform=ccrs.PlateCarree(),
        )
        ax.coastlines()
        ax.add_feature(cartopy.feature.BORDERS, linestyle=":", alpha=1)
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            x_inline=False,
            rotate_labels=False,
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = ticker.FixedLocator([-20, -10, 0, 10, 20, 30])
        ax.set_title(titles[i])
    fig.suptitle(f"{wr}, WRI: p{int(100*quantile)}-p{int(100*(quantile+0.05))}")
    plt.tight_layout()
    plt.savefig(f"{dir_Figures}/{wr}_{int(100*quantile)}.png", dpi=300)
    plt.show()
    composite_dataset.close()


# %%
quantiles = [0.75, 0.90, 0.95]
for i, regime in tqdm(enumerate(CLUSTER_NAMES)):
    for quantile in quantiles:
        composite_wr1 = create_weighted_composite_means(
            df_clusters,
            i,
            quantile,
            PATH_ANOM,
            PATH_PSL,
        )
        plot_composite_meteorology(composite_wr1, regime, quantile)
# %%
