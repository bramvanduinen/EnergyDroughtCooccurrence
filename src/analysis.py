import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap
import matplotlib
import colorcet as cc
import seaborn as sns
import itertools
from tqdm.notebook import tqdm
import pandas as pd
from datetime import datetime, timedelta

ENERGY_PATH = '/net/pc200256/nobackup/users/most/output/LENTIS_PD_02/agg_production/per_country/'

def add_filename(ds, variable):
    # Extract the filename from the 'source' attribute, which is automatically set by open_mfdataset
    filename = os.path.basename(ds.encoding['source'])
    
    # Extract the characters that match the '???' part of the filename
    identifier = filename.split('_')[0]
    
    # Add the identifier as a new coordinate to the dataset
    ds = ds.assign_coords(country_name=identifier)
    if variable == 'sun':
        ds = ds['pv_util'] + ds['pv_roof']
    elif variable == 'wind':
        ds = ds['wind_offshore'] + ds['wind_onshore']
    else:
        ds = ds[variable]

    return ds

def calc_events(data_prod_stack, sorted_indices, num_countries, countries, num_events, dt_event):
    # Initialize empty lists for residuals, runs, and times
    top_vals = np.empty((num_countries, num_events))
    top_times = np.empty((num_countries, num_events), dtype='M8[ns]')
    top_runs = np.empty((num_countries, num_events))

    # Iterate over each country
    for country in range(num_countries):

        eventnr = 0
        begin_ind = 0
        end_ind = 3000 #3000 #initially try 3000 options to find 160 events per country

        while eventnr < num_events:
            # Get the top num_events indices for this country
            top_indices = sorted_indices[country, begin_ind:end_ind].values

            # Get the corresponding residuals, runs, and times
            vals = data_prod_stack[country, top_indices].values
            runs = [int(run[1:]) for run in data_prod_stack.runs[top_indices].values]
            #runs = data_prod_stack.runs[top_indices].values
            times = data_prod_stack.time[top_indices].values

            for i, t in enumerate(times):
                event = (t, runs[i])
                if runs[i] not in top_runs[country,:]:
                    top_vals[country, eventnr] = vals[i]
                    top_times[country, eventnr] = t
                    top_runs[country, eventnr] = runs[i]
                    eventnr += 1
                else:
                    samerun = np.where(runs[i] == top_runs[country,:])[0]
                    dt = abs(t - top_times[country, samerun])
                    if all(dt > np.timedelta64(dt_event, 'D')):
                        dt_d = dt.astype('timedelta64[D]')
                        top_vals[country, eventnr] = vals[i]
                        top_times[country, eventnr] = t
                        top_runs[country, eventnr] = runs[i]
                        eventnr += 1
                if eventnr == num_events:
                    break
            begin_ind = end_ind
            end_ind += 100 #3000

    ds_events = xr.Dataset(
        {
            'vals': (('country', 'event_nr'), top_vals),
            'time': (('country', 'event_nr'), top_times),
            'run': (('country', 'event_nr'), top_runs)
        },
        coords={
            'country': countries,
            'event_nr': np.arange(num_events)
        }
    ) 

    return ds_events

def plot_basics(ds_events, variable, window, dt_event, countries, dir_Figures):
    plt.figure()
    country_ind = 7 #Germany
    plt.scatter(ds_events.time[country_ind], ds_events.vals[country_ind]/1e6, label = countries[country_ind], s = 3)

    country_ind = 12 #France
    plt.scatter(ds_events.time[country_ind], ds_events.vals[country_ind]/1e6, label = countries[country_ind], s = 3)

    country_ind = 13 #United Kingdom
    plt.scatter(ds_events.time[country_ind], ds_events.vals[country_ind]/1e6, label = countries[country_ind], s = 3)

    country_ind = 19 #Italy
    plt.scatter(ds_events.time[country_ind], ds_events.vals[country_ind]/1e6, label = countries[country_ind], s = 3)

    country_ind = 9 #Spain
    plt.scatter(ds_events.time[country_ind], ds_events.vals[country_ind]/1e6, label = countries[country_ind], s = 3)

    # Set the labels and title
    plt.ylabel(f"{variable} [Twh]")
    plt.title(f'1-in-10 yr events')
    plt.legend()
    plt.savefig(f'{dir_Figures}/Events_largecountries_w={window}_dte={dt_event}.png', dpi=300, bbox_inches='tight')

    months_flat = ds_events.time.dt.month.values.flatten()
    #make a histogram of months
    plt.figure()
    counts, edges, bars = plt.hist(months_flat, bins=np.linspace(0.5,12.5, 13))
    plt.xlabel('Month')
    plt.bar_label(bars)
    plt.ylabel('Number of events')
    # make xticks the month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(range(1,13), month_names)
    plt.savefig(f'{dir_Figures}/hist_eventmonths_w={window}_dte={dt_event}.png', dpi=300, bbox_inches='tight')
    plt.close()

    months = ds_events.time.dt.month.values

    #Divide countries into regions of Europe.
    SW = [28, 9, 12, 4, 19]
    NW = [13, 17, 1, 25, 7]
    N = [8, 26, 33, 11]
    C = [27, 0, 32, 31, 6]
    E = [10, 21, 20, 34, 22, 16, 29]
    SE = [2, 5, 14, 24, 30, 3]

    SW_months = np.concatenate([months[i] for i in SW])
    NW_months = np.concatenate([months[i] for i in NW])
    N_months = np.concatenate([months[i] for i in N])
    C_months = np.concatenate([months[i] for i in C])
    E_months = np.concatenate([months[i] for i in E])
    SE_months = np.concatenate([months[i] for i in SE])

    plt.figure()
    counts, edges, bars = plt.hist([SW_months, NW_months, N_months, C_months, E_months, SE_months], 
            bins=np.linspace(0.5, 12.5, 13), stacked=True, 
            label=['SW - 5 countries', 'NW - 5 countries', 'N - 4 countries', 'C - 5 countries', 'E - 7 countries', 'SE - 6 countries'])

    plt.xlabel('Month')
    plt.ylabel('Number of events')
    plt.xticks(range(1,13), month_names)
    plt.legend()
    plt.savefig(f'{dir_Figures}/hist_regionmonths_w={window}_dte={dt_event}.png', dpi=300, bbox_inches='tight')

def calc_cooccurrences(ds_events, num_countries, countries, num_events, dt_cooccur): 
    co_occurrences = np.zeros((num_countries, num_countries), dtype=float)
    df_cooccurrences_list = []

    for country_1, country_2 in itertools.product(ds_events.country.values, repeat=2):
        # Precompute selections
        id_country_1 = np.where(countries == country_1)[0][0]
        id_country_2 = np.where(countries == country_2)[0][0]
        run_country_1 = ds_events.run.sel(country=country_1).values
        run_country_2 = ds_events.run.sel(country=country_2).values
        time_country_1 = ds_events.time.sel(country=country_1).values
        time_country_2 = ds_events.time.sel(country=country_2).values

        for i in range(num_events):
            dt = np.abs(time_country_2 - time_country_1[i])
            timeok = dt <= np.timedelta64(dt_cooccur, 'D')
            
            run_1 = run_country_1[i]
            run_2 = run_country_2[timeok]

            sum_run = np.sum(run_1 == run_2)
            co_occurrences[id_country_1, id_country_2] += sum_run 

            if sum_run > 0:
                for j in range(sum_run):
                    # Combine the conditions
                    condition = (timeok) & (run_1 == run_country_2)
                    times = time_country_2[condition]
                    # Get the indices of non-NaN values
                    non_nan_indices = np.argwhere(~pd.isnull(times)).flatten()
                    jth_non_nat = times[non_nan_indices[j]]                
                    df_cooccurrences_list.append({'country_1': country_1, 'country_2': country_2, 
                                                    'date_1': time_country_1[i], 
                                                    'date_2': jth_non_nat, 
                                                    'run': run_1})
    # Convert list of dicts to DataFrame
    df_cooccurrences = pd.DataFrame(df_cooccurrences_list)

    # Convert to xarray
    co_occurrences = xr.DataArray(
    co_occurrences,
    coords={'country_1': countries, 'country_2': countries},
    dims=['country_1', 'country_2']
    )
    
    return co_occurrences, df_cooccurrences

def find_non_producers(co_occurrences, data_prod, homedir):
    countries = data_prod.country.values
    country_names = data_prod.country_name.values
    row_order = np.load(f'{homedir}Data/row_order_residual.npy') #load the row ordering of the clustered residual heatmap, to follow the same clustering!
    ordered_countries = countries[row_order]
    ordered_countrynames = country_names[row_order]

    co_occurrences = co_occurrences.isel(country_1 = row_order, country_2 = row_order) # use same order always to sort co_occurrences matrix

    onerun = data_prod.isel(runs = 45) # 45 is a random choice, for speed only check for one run.
    noprod = []

    for i, c in enumerate(ordered_countries):
        totalprod = np.sum(onerun.sel(country = c))
        if totalprod == 0:
            noprod.append(int(i))
    if len(noprod) > 0:
        print(f'The countries that do not produce are {ordered_countrynames[noprod]}')

    co_occurrences[noprod] = np.nan
    co_occurrences[:, noprod] = np.nan
    return co_occurrences

def plot_heatmap(co_occurrences, countries, window, dt_event, dt_cooccur, dir_Figures, homedir):
    row_order = np.load(f'{homedir}Data/row_order_residual.npy') #load the row ordering of the clustered residual heatmap, to follow the same clustering!
    # co_occurrences_plot = co_occurrences.isel(country_1 = row_order, country_2 = row_order)
    co_occurrences_plot = co_occurrences.copy()
    np.fill_diagonal(co_occurrences_plot.values, np.nan) #fill main diagonal with nans, countries cannot co-occur with themselves
    #vmax_val = np.max(co_occurrences_plot).values + 1

    vmax_val = 120 #set vmax to 120 (based on 160 events), so that the colorbar is the same for all plots

    plt.figure(figsize=(10, 8))
    cmap = matplotlib.colormaps["Reds"]
    cmap.set_bad(color='0.8')

    ax = sns.heatmap(co_occurrences_plot, cmap=cmap, xticklabels=countries[row_order], yticklabels=countries[row_order], vmax=vmax_val, cbar = False)
    cbar = plt.colorbar(ax.collections[0], extend='max')

    plt.savefig(f'{dir_Figures}/clusteredheatmap_co_occurrence_w={window}_dte={dt_event}_dtc={dt_cooccur}.png', dpi=300, bbox_inches='tight')

def plot_clustermap(co_occurrences, countries, rundate):
    co_occurrences_plot = co_occurrences.copy()
    np.fill_diagonal(co_occurrences_plot.values, np.nan) # fill main diagonal with nans, countries cannot co-occur with themselves

    max_co_oc = np.max(co_occurrences_plot).values + 1 # add one so that maximum is not grey

    # Create a colormap that maps main diagonal to grey (above the maximum co-occurrence between different countries) and the rest of the values to the original colormap
    cmap = matplotlib.colormaps["Reds"]
    cmap.set_over('0.8')

    # Plot the clustermap with the modified colormap and vmin set to 0
    cg = sns.clustermap(co_occurrences, cmap=cmap, method='ward', xticklabels=countries, yticklabels=countries, vmax=max_co_oc)
    cg.ax_row_dendrogram.set_visible(False)
    cg.ax_col_dendrogram.set_visible(False)
    x0, _y0, _w, _h = cg.cbar_pos
    cg.ax_cbar.set_position([1, 0.06, 0.025, 0.74])
    # plt.savefig(f'{dir_Figures}/clusteredheatmap_co_occurrence_w={WINDOW}_dte={DT_EVENT}_dtc={DT_COOCCUR}.png', dpi=300, bbox_inches='tight')

    # only had to be done once for the residual heatmap. Rest is not clustered by sns anymore, but follows that clustering
    row_order = cg.dendrogram_row.reordered_ind
    np.save(f'../Data/row_order_residual_v{rundate}.npy', row_order) 

def save_data(co_occurrences, df_cooccurrences, window, rundate, dir_Results):
    np.save(f'{dir_Results}/co_occurrences_w={window}_{rundate}.npy', co_occurrences.values)
    df_cooccurrences.to_csv(f'{dir_Results}/df_co_occurrences_w={window}_{rundate}.csv', index=False)
    print("Calculated data is saved")

def analysis(variable, num_events, window, dt_event, dt_cooccur):
    print(f'Starting analysis for {variable}')

    date_now = datetime.now()
    rundate = '%4.4i%2.2i%2.2i' % (date_now.year,date_now.month,date_now.day)

    if variable == 'wind' or variable == 'sun':
        variabletype = 'supply' # changes direction of sorting. For supply you want minima, for demand you want maxima
    if variable == 'residual' or variable == 'demand':
        variabletype = 'demand' # changes direction of sorting. For supply you want minima, for demand you want maxima
    
    homedir = '/usr/people/duinen/MSc-thesis/'
    # Create folders for Figures from today
    dir_Figures = f'{homedir}Results/Figures/{rundate}/{variable}'

    if not os.path.exists(dir_Figures):
        print('Creating dir %s' % dir_Figures)
        os.makedirs(dir_Figures)

    dir_Results = f'{homedir}Results/Data/{rundate}/{variable}'

    if not os.path.exists(dir_Results):
        print('Creating dir %s' % dir_Results)
        os.makedirs(dir_Results)

    data_prod = xr.open_mfdataset(ENERGY_PATH + '???' + '_LENTIS_PD_02_v4.nc', combine='nested', concat_dim='country', preprocess=lambda ds: add_filename(ds, variable))
    data_prod = data_prod.drop_sel(country = [0, 23, 28, 38]) # drop countries that are not properly represented in the analysis

    num_countries = np.shape(data_prod)[0]
    print(f"The number of analyzed countries is {num_countries}.")

    data_prod_ma = data_prod.rolling(time=window).mean() # rolling average over window
    countries = data_prod_ma.country_name.values

    data_prod_stack = data_prod_ma.stack(event=('time', 'runs'))
    if variabletype == 'supply': #for supply an event is a minimum
        sorted_indices = data_prod_stack.compute().argsort()
    elif variabletype == 'demand': #for demand an event is a maximum
        sorted_indices = (-data_prod_stack.compute()).argsort()

    ds_events = calc_events(data_prod_stack, sorted_indices, num_countries, countries, num_events, dt_event)
    plot_basics(ds_events, variable, window, dt_event, countries, dir_Figures)
    co_occurrences, df_cooccurrences = calc_cooccurrences(ds_events, num_countries, countries, num_events, dt_cooccur)
    co_occurrences = find_non_producers(co_occurrences, data_prod, homedir)
    plot_heatmap(co_occurrences, countries, window, dt_event, dt_cooccur, dir_Figures, homedir)
    save_data(co_occurrences, df_cooccurrences, window, rundate, dir_Results)

def analysis_clustermap(variable, num_events, window, dt_event, dt_cooccur):
    print(f'Starting analysis for {variable}')

    date_now = datetime.now()
    rundate = '%4.4i%2.2i%2.2i' % (date_now.year,date_now.month,date_now.day)

    if variable == 'wind' or variable == 'sun':
        variabletype = 'supply' # changes direction of sorting. For supply you want minima, for demand you want maxima
    if variable == 'residual' or variable == 'demand':
        variabletype = 'demand' # changes direction of sorting. For supply you want minima, for demand you want maxima
    
    homedir = '/usr/people/duinen/MSc-thesis/'
    # Create folders for Figures from today
    dir_Figures = f'{homedir}Results/Figures/{rundate}/{variable}'

    if not os.path.exists(dir_Figures):
        print('Creating dir %s' % dir_Figures)
        os.makedirs(dir_Figures)

    dir_Results = f'{homedir}Results/Data/{rundate}/{variable}'

    if not os.path.exists(dir_Results):
        print('Creating dir %s' % dir_Results)
        os.makedirs(dir_Results)

    data_prod = xr.open_mfdataset(ENERGY_PATH + '???' + '_LENTIS_PD_02_v4.nc', combine='nested', concat_dim='country', preprocess=lambda ds: add_filename(ds, variable))
    data_prod = data_prod.drop_sel(country = [0, 23, 28, 38]) # drop countries that are not properly represented in the analysis

    num_countries = np.shape(data_prod)[0]
    print(f"The number of analyzed countries is {num_countries}.")

    data_prod_ma = data_prod.rolling(time=window).mean() # rolling average over window
    countries = data_prod_ma.country_name.values

    data_prod_stack = data_prod_ma.stack(event=('time', 'runs'))
    if variabletype == 'supply': #for supply an event is a minimum
        sorted_indices = data_prod_stack.compute().argsort()
    elif variabletype == 'demand': #for demand an event is a maximum
        sorted_indices = (-data_prod_stack.compute()).argsort()

    ds_events = calc_events(data_prod_stack, sorted_indices, num_countries, countries, num_events, dt_event)
    plot_basics(ds_events, variable, window, dt_event, countries, dir_Figures)
    co_occurrences, df_cooccurrences = calc_cooccurrences(ds_events, num_countries, countries, num_events, dt_cooccur)
    # co_occurrences = find_non_producers(co_occurrences, data_prod, homedir)
    
    co_occurrences_plot = co_occurrences.copy()
    np.fill_diagonal(co_occurrences_plot.values, np.nan) # fill main diagonal with nans, countries cannot co-occur with themselves

    max_co_oc = np.max(co_occurrences_plot).values + 1 # add one so that maximum is not grey

    # Create a colormap that maps main diagonal to grey (above the maximum co-occurrence between different countries) and the rest of the values to the original colormap
    cmap = matplotlib.colormaps["Reds"]
    cmap.set_over('0.8')

    # Plot the clustermap with the modified colormap and vmin set to 0
    cg = sns.clustermap(co_occurrences, cmap=cmap, method='ward', xticklabels=countries, yticklabels=countries, vmax=max_co_oc)
    cg.ax_row_dendrogram.set_visible(False)
    cg.ax_col_dendrogram.set_visible(False)
    x0, _y0, _w, _h = cg.cbar_pos
    cg.ax_cbar.set_position([1, 0.06, 0.025, 0.74])
    # plt.savefig(f'{dir_Figures}/clusteredheatmap_co_occurrence_w={WINDOW}_dte={DT_EVENT}_dtc={DT_COOCCUR}.png', dpi=300, bbox_inches='tight')

    # only had to be done once for the residual heatmap. Rest is not clustered by sns anymore, but follows that clustering
    row_order = cg.dendrogram_row.reordered_ind
    print(row_order)
    np.save(f'../Data/row_order_residual_v{rundate}.npy', row_order) 
    # plot_clustermap(co_occurrences, countries, rundate)
    # save_data(co_occurrences, df_cooccurrences, window, rundate, dir_Results)