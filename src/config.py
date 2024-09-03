# Paths
PATH_CLUSTERS = "/usr/people/duinen/MSc-thesis/Results/clusters/"
PATH_DATA = "/usr/people/duinen/MSc-thesis/Results/data/"
PATH_ED = "/usr/people/duinen/MSc-thesis/src/find_energydroughts/data/"
PATH_ED_REGIONS = "/usr/people/duinen/MSc-thesis/src/find_energydroughts_region/data/"
PATH_ANOM = "/net/pc230050/nobackup/users/duinen/LENTIS/present/"
PATH_ZG500 = "/net/pc230050/nobackup/users/duinen/LENTIS/present/zg500_d/"
PATH_ZG500_ERA5 = "/net/pc230050/nobackup/users/duinen/LENTIS/ERA5/"
PATH_PSL = "/net/pc200256/nobackup/users/most/LENTIS/present/day/psl_d"
ED_FILENAME_COUNTRIES = "netto_demand_el7_winter_LENTIS_2023_PD_1600_events.csv"
ED_FILENAME_REGIONS = "max_drought_regions_netto_demand_el7_winter_LENTIS_2023_PD_1600_events.csv"
ED_FILENAME_REGIONS_RANDOM = "random_regions_netto_demand_el7_winter_LENTIS_2023_PD_1600_events.csv"

# constants
CLUSTER_NAMES = ["NAO +", "NAO \u2212", "Blocking", "Atl. Ridge"]
WINDOW = 7

# clustering parameters
VERSION = "Bayes_full_v2"  # update this when you change the clustering algorithm and want to save the new results
USE_SCALING = 0
N_EOFS = 20
N_EOFS_FOR_KMEANS = 20
N_CLUSTERS = 4
THRES = 0.8  # If posterior probability of best-regime is below this threshold, assign to no regime

# energy drought parameters
EVENTTYPE = "random"
EVENT_LENGTH = 7
NUM_EVENTS = 1600
COUNTRIES = [
    "AUT",
    "BEL",
    "CHE",
    "CZE",
    "DEU",
    "DNK",
    "ESP",
    "EST",
    "FIN",
    "FRA",
    "GBR",
    "HRV",
    "HUN",
    "IRL",
    "ITA",
    "LTU",
    "LVA",
    "NLD",
    "NOR",
    "POL",
    "PRT",
    "SVK",
    "SVN",
    "SWE",
]

REGIONS = {
    "IB": {"color": "C4", "countries": ["PRT", "ESP"]},
    "NW": {"color": "C1", "countries": ["NLD", "DEU", "BEL", "DNK", "GBR", "IRL"]},
    "N": {"color": "C0", "countries": ["NOR", "SWE", "FIN"]},
    "B": {"color": "cyan", "countries": ["EST", "LVA", "LTU"]},
    "C": {"color": "C2", "countries": ["AUT", "ITA", "CHE", "FRA"]},
    "E": {"color": "C3", "countries": ["POL", "HRV", "HUN", "SVN", "CZE", "SVK"]},
}
