# import os
# import xarray as xr
# import matplotlib.pyplot as plt
# import numpy as np
# import cartopy.crs as ccrs
# from matplotlib.colors import ListedColormap
# import matplotlib
# import colorcet as cc
# import seaborn as sns
# import itertools
# from tqdm.notebook import tqdm
# import pandas as pd
# from datetime import datetime, timedelta

from analysis import analysis

ENERGY_PATH = '/net/pc200256/nobackup/users/most/output/LENTIS_PD_02/agg_production/per_country/'

#variable (str): 'residual', 'demand', 'wind', 'sun'
#num_events (int): number of events to analyze (160 is 1-in-10 yr event)
#window (int): window size in days, for rolling mean
#dt_event (int): minimum number of days in-between consecutive events
#dt_cooccur (int): maximum number of days in-between co-occurrences between countries

analysis(variable = 'residual', num_events = 160, window = 7, dt_event = 14, dt_cooccur = 14)
analysis(variable = 'demand', num_events = 160, window = 7, dt_event = 14, dt_cooccur = 14)
analysis(variable = 'wind', num_events = 160, window = 7, dt_event = 14, dt_cooccur = 14)
analysis(variable = 'sun', num_events = 160, window = 7, dt_event = 14, dt_cooccur = 14)