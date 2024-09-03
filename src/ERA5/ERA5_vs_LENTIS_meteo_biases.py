# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cdo import *
from matplotlib import colors

cdo = Cdo()

# %%
ANOM_PATH = "/net/pc230050/nobackup/users/duinen/LENTIS/present/"
PSL_PATH = "/net/pc200256/nobackup/users/most/LENTIS/present/day/psl_d/"
ERA_PATH = "/net/pc230050/nobackup/users/duinen/LENTIS/ERA5/"
LENTIS_PATH = "/net/pc200256/nobackup/users/most/LENTIS/present/"
RUNS = np.arange(10, 170)

temp_era5 = xr.open_dataset(ERA_PATH + "era5_t2m_d_1950_2022.nc")["t2m"]
wind_era5 = xr.open_dataset(ERA_PATH + "era5_sfcWind_d_1950_2022.nc")["sfcWind"]
solar_era5 = xr.open_dataset(
    "/net/pc200256/nobackup/users/most/ERA5/voorBram/era5_ssrd_d_1950_2022_shifted11h.nc",
)["ssrd"]

# WIND_MERRA_FILE = (
#     "/net/pc200256/nobackup/users/most/reanalysis_m/merra_wind/wind_m_hourlybased_1990-2019.nc"
# )
ERA_PR_FILE = ERA_PATH + "era5_pr_monthly_1950_2024_eragrid.nc"
ERA_PR_REGRID = ERA_PATH + "era5_pr_monthly_1950_2024_lentisgrid.nc"
WIND_MERRA_REGRID = "/net/pc230050/nobackup/users/duinen/MERRA/ws_merra_1990_2019_lentisgrid.nc"

# only had to be done once, so commented out
# cdo.remapcon(
#     "/net/pc200256/nobackup/users/most/LENTIS/LENTIS_gridarea.nc",
#     input=ERA_PR_FILE,
#     output=ERA_PR_REGRID,
# )

ws_merra = xr.open_dataset(WIND_MERRA_REGRID)["wind_speed"]
pr_era = xr.open_dataset(ERA_PR_REGRID, decode_times=False)["tp"]

temp_era5 = temp_era5.sel(time=slice("1990", "2019"))  # select PD climatology of ERA5
wind_era5 = wind_era5.sel(time=slice("1990", "2019"))  # select PD climatology of ERA5
solar_era5 = solar_era5.sel(time=slice("1990", "2019"))  # select PD climatology of ERA5
pr_era5 = pr_era.sel(time=slice("480", "840"))  # select PD climatology of ERA5 (1990-2019)

# %%
era5_clim_month = temp_era5.groupby("time.month").mean("time")
lentis_clim_month = xr.open_dataset(LENTIS_PATH + "climatology/tas_m_ymonmean.nc")["tas"]
lentis_clim_month = lentis_clim_month.assign_coords(
    month=lentis_clim_month["time"].dt.month,
).swap_dims({"time": "month"})


# %%
lentis_wind_day = xr.open_dataset(LENTIS_PATH + "climatology/sfcWind_d_ydaymean.nc")["sfcWind"]
lentis_wind_month = (
    lentis_wind_day.assign_coords(
        month=lentis_wind_day["time"].dt.month,
    )
    .groupby("month")
    .mean()
)

era5_wind_month = wind_era5.groupby("time.month").mean("time")

ws_merra_month = ws_merra.assign_coords(month=ws_merra["time"].dt.month).groupby("month").mean()

# %%
lentis_solar_day = xr.open_dataset(LENTIS_PATH + "climatology/rsds_d_ydaymean.nc")["rsds"]
lentis_solar_month = (
    lentis_solar_day.assign_coords(
        month=lentis_solar_day["time"].dt.month,
    )
    .groupby("month")
    .mean()
)

era5_solar_month = solar_era5.groupby("time.month").mean("time")


# %%
lentis_precip_day = xr.open_dataset(
    "/net/pc230050/nobackup/users/duinen/LENTIS/present/pr_d_anomaly/pr_clim_ensmean_7d.nc",
)["pr"]
lentis_precip_month = (
    lentis_precip_day.assign_coords(
        month=lentis_precip_day["time"].dt.month,
    )
    .groupby("month")
    .mean()
)

# HACK: ERA5 monthly unit is in "months since 1950-01", and conversion didn't work out easily.
# so just select the months nov-march manually
mask_nov_mar = np.isin(pr_era5.time % 12, [0, 1, 2, 10, 11])
era5_precip_clim = pr_era5.sel(time=mask_nov_mar).mean(dim="time", skipna=True)

# %%
months = [1, 2, 3, 11, 12]

# %% Seasonally averaged biases:
lon_min, lon_max, lat_min, lat_max = -25, 30, 35, 72

diff_temp_mean = lentis_clim_month.sel(
    month=months,
).mean(dim="month") - temp_era5.sel(time=temp_era5["time.month"].isin(months)).mean(dim="time")

diff_solar_mean = lentis_solar_month.sel(
    month=months,
).mean(dim="month") - era5_solar_month.sel(month=months).mean(dim="month")

diff_wind_mean = lentis_wind_month.sel(
    month=months,
).mean(dim="month") - wind_era5.sel(time=wind_era5["time.month"].isin(months)).mean(dim="time")

diff_wind_mean_merra = lentis_wind_month.sel(
    month=months,
).mean(dim="month") - ws_merra_month.sel(month=months).mean(dim="month")

# convert lentis precip to mm/day
diff_precip_mean = (
    lentis_precip_month.sel(
        month=months,
    ).mean(dim="month")
    * 86400
) - era5_precip_clim
# %% PLOT WIND TOGETHER, PAPER FIGURE

fontsize = 20

fig, axs = plt.subplots(
    1,
    2,
    figsize=(20, 10),
    dpi=300,
    subplot_kw={"projection": ccrs.PlateCarree()},
)

# Plot for LENTIS - ERA5
axs[0].set_extent([-12, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
axs[0].pcolormesh(
    diff_wind_mean.lon,
    diff_wind_mean.lat,
    diff_wind_mean,
    cmap=plt.cm.PiYG,
    vmin=-3,
    vmax=3,
    transform=ccrs.PlateCarree(),
)
axs[0].coastlines()
axs[0].set_title("(a) LENTIS - ERA5", loc="left", fontsize=fontsize)

# Plot for LENTIS - MERRA
axs[1].set_extent([-12, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
axs[1].pcolormesh(
    diff_wind_mean_merra.lon,
    diff_wind_mean_merra.lat,
    diff_wind_mean_merra,
    cmap=plt.cm.PiYG,
    vmin=-3,
    vmax=3,
    transform=ccrs.PlateCarree(),
)
axs[1].coastlines()
axs[1].set_title("(b) LENTIS - MERRA-2", loc="left", fontsize=fontsize)
cbar = fig.colorbar(
    axs[1].collections[0],
    ax=axs,
    orientation="vertical",
    fraction=0.015,
    pad=0.05,
    extend="both",
)
cbar.set_label("[m/s]", fontsize=fontsize - 2)
cbar.ax.tick_params(labelsize=fontsize - 2)

# %% Plot biases of temperature, solar radiation and precipitation

# cmap_solar
GyYl = colors.LinearSegmentedColormap.from_list(
    "my_colormap",
    [(0, "grey"), (0.5, "white"), (1, "orange")],  # (position, color)
)

# Create a figure with one row and three columns
fig, axs = plt.subplots(
    1,
    3,
    figsize=(15, 5),
    dpi=300,
    subplot_kw={"projection": ccrs.PlateCarree()},
)

# Temperature plot
axs[0].set_extent([-12, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
temp_cm = axs[0].pcolormesh(
    diff_temp_mean.lon,
    diff_temp_mean.lat,
    diff_temp_mean,
    cmap="RdBu_r",
    vmin=-4,
    vmax=4,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(temp_cm, ax=axs[0], shrink=0.6, extend="both")
cbar.set_label("[K]", fontsize=fontsize - 4)
cbar.ax.tick_params(labelsize=fontsize - 4)
axs[0].coastlines()
axs[0].set_title("(a) 2 m temperature", loc="left", fontsize=fontsize)

# Solar radiation plot
axs[1].set_extent([-12, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
solar_cm = axs[1].pcolormesh(
    diff_solar_mean.lon,
    diff_solar_mean.lat,
    diff_solar_mean,
    cmap=GyYl,
    vmin=-15,
    vmax=15,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(solar_cm, ax=axs[1], shrink=0.6, extend="min")
cbar.set_label("[W/m\u00b2]", fontsize=fontsize - 4)
cbar.ax.tick_params(labelsize=fontsize - 4)
axs[1].coastlines()
axs[1].set_title("(b) Incoming solar radiation", loc="left", fontsize=fontsize)

# Precipitation plot
axs[2].set_extent([-12, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
precip_cm = axs[2].pcolormesh(
    diff_precip_mean.lon,
    diff_precip_mean.lat,
    diff_precip_mean,
    cmap=plt.cm.BrBG,
    vmin=-2,
    vmax=2,
    transform=ccrs.PlateCarree(),
)
cbar = fig.colorbar(precip_cm, ax=axs[2], shrink=0.6, extend="both")
cbar.set_label("[mm/day]", fontsize=fontsize - 4)
cbar.ax.tick_params(labelsize=fontsize - 4)
axs[2].coastlines()
axs[2].set_title("(c) Precipitation", loc="left", fontsize=fontsize)

plt.tight_layout()

# %%
