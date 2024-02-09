import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

def plot_maxmin_points(ax, lons, lats, lon_lim, lat_lim, data, extrema, nsize, symbol, color='k',
                       plotValue=True, transform=None):
    """
    Plot relative maximum and minimum points for a 2D grid. The function
    can be used to plot an H for maximum values (e.g., High pressure) and an L for minimum
    values (e.g., low pressue). It is best to used filetered data to obtain  a synoptic scale
    max/min value. The symbol text can be set to a string value and optionally the color of the
    symbol and any plotted value can be set with the parameter color

    Parameters:
        ax (matplotlib.axes.Axes): Matplotlib axes to plot on.
        lons (numpy.ndarray): 1D array of plotting longitude values.
        lats (numpy.ndarray): 1D array of plotting latitude values.
        lon_lim (float): Maximum longitude extent for plotting.
        lat_lim (float): Maximum latitude extent for plotting.
        data (numpy.ndarray): 2D data to find max/min values and plot symbols.
        extrema (str): Either 'max' for Maximum Values or 'min' for Minimum Values.
        nsize (int): Size of the grid box to filter max/min values for reasonable plotting.
        symbol (str): String to be placed at the location of max/min value.
        color (str, optional): Matplotlib color name to plot the symbol (and numerical value, if plotted).
        plotValue (bool, optional): Whether to plot the numeric value of max/min point.
        transform (matplotlib.transforms.Transform, optional): Matplotlib transform for symbol placement.

    Returns:
        None

    Notes:
        - #REF: Adapted from: https://unidata.github.io/python-gallery/examples/HILO_Symbol_Plot.html.
            BSD 3-Clause License
            Copyright (c) 2019, Unidata
            All rights reserved.
    """
    if (extrema == 'max'):
        data_ext = maximum_filter(data, nsize, mode='nearest')
    elif (extrema == 'min'):
        data_ext = minimum_filter(data, nsize, mode='nearest')
    else:
        raise ValueError('Value for hilo must be either max or min')

    mxy, mxx = np.where(data_ext == data)

    lon, lat = np.meshgrid(lons[lons < lon_lim], lats[lats < lat_lim])

    for i in range(len(mxy)):
        try:
            ax.text(lon[mxy[i], mxx[i]], lat[mxy[i], mxx[i]], symbol, color=color, size=24,
                    clip_on=True, horizontalalignment='center', verticalalignment='center',
                    transform=transform)
            if plotValue:
                ax.text(lon[mxy[i], mxx[i]], lat[mxy[i], mxx[i]],
                        '\n' + str(int(data[mxy[i], mxx[i]])),
                        color=color, size=12, clip_on=True, fontweight='bold',
                        horizontalalignment='center', verticalalignment='top', transform=transform)
        except IndexError:
            continue