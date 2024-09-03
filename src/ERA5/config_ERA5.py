# Paths
PATH_CLUSTERS = "/usr/people/duinen/MSc-thesis/Results/clusters/"
PATH_DATA = "/usr/people/duinen/MSc-thesis/Results/data/"
PATH_ED = "/usr/people/duinen/MSc-thesis/src/find_energydroughts/data/"
PATH_ED_REGION = "/usr/people/duinen/MSc-thesis/src/find_energydroughts_region/data/"
PATH_ANOM = "/net/pc230050/nobackup/users/duinen/LENTIS/present/zg500_anomaly/"
PATH_ZG500 = "/net/pc230050/nobackup/users/duinen/LENTIS/present/zg500_d/"
PATH_ZG500_ERA5 = "/net/pc230050/nobackup/users/duinen/LENTIS/ERA5/"

# constants
CLUSTER_NAMES = ["NAO +", "NAO \u2212", "Blocking", "Atl. Ridge"]
# CLUSTER_NAMES = ["Blocking", "NAO -", "NAO +", "Atl. Ridge"] # changed 2024-03-15 to match new reordering script
WINDOW = 7

# clustering parameters
VERSION = "ERA5_v1"  # update this when you change the clustering algorithm and want to save the new results
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
    "IB": ["PRT", "ESP"],
    "NW": ["NLD", "DEU", "BEL", "DNK", "GBR", "IRL"],
    "N": ["NOR", "SWE", "FIN"],
    "B": ["EST", "LVA", "LTU"],
    "C": ["AUT", "ITA", "CHE", "FRA"],
    "E": ["POL", "HRV", "HUN", "SVN", "CZE", "SVK"],
}

# Lieke
FOLDER = "/net/pc200256//nobackup/users/most/output/LENTIS_2023_PD/agg_production/per_country/"
RUNNAME = "LENTIS_2023_PD"
OFOLDER = "data/"
