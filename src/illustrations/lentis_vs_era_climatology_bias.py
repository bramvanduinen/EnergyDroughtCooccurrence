# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cdo import *

cdo = Cdo()

# %%
ANOM_PATH = "/net/pc230050/nobackup/users/duinen/LENTIS/present/"
PSL_PATH = "/net/pc200256/nobackup/users/most/LENTIS/present/day/psl_d/"
ERA_PATH = "/net/pc230050/nobackup/users/duinen/LENTIS/ERA5/"
LENTIS_PATH = "/net/pc200256/nobackup/users/most/LENTIS/present/"
RUNS = np.arange(10, 170)

temp_era5 = xr.open_dataset(ERA_PATH + "era5_t2m_d_1950_2022.nc")["t2m"]
wind_era5 = xr.open_dataset(ERA_PATH + "era5_sfcWind_d_1950_2022.nc")["sfcWind"]

# cdo.detrend(
#     input=ERA_PATH + "era5_t2m_d_1950_2022.nc", output=ERA_PATH + "era5_t2m_d_1950_2022_detrend.nc"
# )

# temp_era5_detrend = xr.open_dataset(ERA_PATH + "era5_t2m_d_1950_2022_detrend.nc")["t2m"]
temp_era5 = temp_era5.sel(time=slice("1990", "2019"))  # select PD climatology of ERA5
wind_era5 = wind_era5.sel(time=slice("1990", "2019"))  # select PD climatology of ERA5
# for run in tqdm(RUNS):
#     temp_lentis_run = xr.open_dataset(
#         ANOM_PATH + f"tas_d_anomaly/anom_tas_d_ECEarth3_h{run:03d}.nc",
#     )["tas"]
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

# %%
months = [1, 2, 3, 11, 12]

for month in months:
    diff_temp = lentis_clim_month.sel(
        month=month,
    ) - temp_era5.sel(time=temp_era5["time.month"] == month)
    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-25, 55, 35, 71])
    diff_temp.mean(dim="time").plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        vmin=-3,
        vmax=3,
    )
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    plt.title(f"T2m LENTIS - ERA5, month = {month}")


# %%
for month in months:
    diff_wind = lentis_wind_month.sel(
        month=month,
    ) - wind_era5.sel(time=wind_era5["time.month"] == month)
    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-25, 55, 35, 71])
    diff_wind.mean(dim="time").plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        vmin=-3,
        vmax=3,
    )
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    plt.title(f"Wind LENTIS - ERA5, month = {month}")

# %% Seasonally averaged biases:
lon_min, lon_max, lat_min, lat_max = -25, 30, 35, 72

diff_temp_mean = lentis_clim_month.sel(
    month=months,
).mean(dim="month") - temp_era5.sel(time=temp_era5["time.month"].isin(months)).mean(dim="time")
plt.figure(dpi=300)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
diff_temp_mean.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap="RdBu_r",
    vmin=-4,
    vmax=4,
    cbar_kwargs={"label": "[K]", "shrink": 0.8},
)
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="black")
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
plt.title("T2m, LENTIS - ERA5, NDJFM")

diff_wind_mean = lentis_wind_month.sel(
    month=months,
).mean(dim="month") - wind_era5.sel(time=wind_era5["time.month"].isin(months)).mean(dim="time")
plt.figure(dpi=300)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
diff_wind_mean.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap="RdBu_r",
    vmin=-3,
    vmax=3,
    cbar_kwargs={"label": "[m/s]", "shrink": 0.8},
)

ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="black")
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
plt.title("Wind speed (10m), LENTIS - ERA5, NDJFM")


# %%
