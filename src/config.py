# Paths
PATH_CLUSTERS = "/usr/people/duinen/MSc-thesis/Results/clusters/"
PATH_ED = "/usr/people/duinen/MSc-thesis/src/energydroughts-Europe/data/"
PATH_ANOM = "/net/pc230050/nobackup/users/duinen/LENTIS/present/zg500_anomaly/"
PATH_ZG500 = "/net/pc230050/nobackup/users/duinen/LENTIS/present/zg500_d/"

# constants
CLUSTER_NAMES = ["NAO +", "NAO -", "Blocking", "Atl. Ridge"]
# CLUSTER_NAMES = ["Blocking", "NAO -", "NAO +", "Atl. Ridge"] # changed 2024-03-15 to match new reordering script
WINDOW = 7

# clustering parameters
VERSION = "Bayes_full_v2"  # update this when you change the clustering algorithm and want to save the new results
USE_SCALING = 0
N_EOFS = 20
N_EOFS_FOR_KMEANS = 20
N_CLUSTERS = 4
THRES = 0.8  # If posterior probability of best-regime is below this threshold, assign to no regime


# Lieke
FOLDER = "/net/pc200256//nobackup/users/most/output/LENTIS_2023_PD/agg_production/per_country/"
RUNNAME = "LENTIS_2023_PD"
OFOLDER = "data/"
