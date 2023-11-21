import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from netCDF4 import Dataset
import time
import swifter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
sns.set_theme()
sns.set_palette(palette = "rainbow")
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer

sys.path.append('..')
from general_functions_ccvi.normality_test import normality_test
from general_functions_ccvi.log_with_pop_and_fit_normal_distribution import custom_norm


def get_LC_class_4_apply(row):

    # Open the NetCDF file
    nc = Dataset('/DATA/REFERENCE_DATASETS/LANDCOVER/C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc', 'r')

    # Specify the target latitude and longitude
    target_lat = row["latitude"]
    target_lon = row["longitude"]

    # Read latitude and longitude values from the NetCDF file
    lats = nc.variables['lat'][:]
    lons = nc.variables['lon'][:]

    lat_index = np.argmin(np.abs(lats - target_lat))
    lon_index = np.argmin(np.abs(lons - target_lon))

    lc_class = nc.variables['lccs_class'][:, lat_index, lon_index]

    nc.close()

    return int(lc_class)

def get_LC_class(target_lat,target_lon):

    # Open the NetCDF file
    nc = Dataset('/DATA/REFERENCE_DATASETS/LANDCOVER/C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc', 'r')

    # Specify the target latitude and longitude
    #target_lat = row["latitude"]
    #target_lon = row["longitude"]

    # Read latitude and longitude values from the NetCDF file
    lats = nc.variables['lat'][:]
    lons = nc.variables['lon'][:]

    lat_index = np.argmin(np.abs(lats - target_lat))
    lon_index = np.argmin(np.abs(lons - target_lon))

    lc_class = nc.variables['lccs_class'][:, lat_index, lon_index]

    nc.close()

    return int(lc_class)

def to_bin(x):
    binsize = 0.5

    if x > 180:
        x = -180 + x - 180
    return (binsize / 2) + np.floor(x / binsize) * binsize

def fire_per_ab(df,col,population):

    df = df.merge(popualtion[["pgid","wp_pop_density"]],on="pgid",how="left")
    new_name="fire_per_pop_density"
    df[new_name] = df[col] / df["wp_pop_density"]
    return df



def process_data(rootdirsearch,priogrid, popualtion):
    reference_pgid = priogrid
    df_list = []
    for path, subdirs, files in os.walk(rootdirsearch):
        for name in files:
            if name.endswith("csv"):
                print(name)

                df = pd.read_csv(f"{path}/{name}")

                #check if the point is not in classes 10,11,12,20,30,40

                df["latbin"] = df.latitude.map(to_bin)
                df["lonbin"] = df.longitude.map(to_bin)

                if not os.path.exists(f"{path}/{name.replace('.csv','.csvx')}"):
                    start_time = time.perf_counter()
                    df["LC"] = df.swifter.apply(get_LC_class_4_apply,axis=1)
                    end_time = time.perf_counter()
                    elapsed_time = end_time - start_time
                    print("Elapsed time: ", elapsed_time)

                    df.to_csv(f"{path}/{name.replace('.csv','.csvx')}")
                else:
                    df =pd.read_csv(f"{path}/{name.replace('.csv','.csvx')}")


                df["count_fire"] = 0
                df.loc[((df.confidence > 95) & (~(df.LC.isin([10,11,12,20,30,40])))), "count_fire"] = 1



                df = df.merge(reference_pgid, left_on=["latbin", "lonbin"], right_on=["lat", "lon"], how="left")

                df = df[["pgid", "acq_date", "lat", "lon","count_fire"]]

                df = df.groupby(by=["pgid", "acq_date", "lat", "lon"]).sum().reset_index()
                df['quarter'] = pd.to_datetime(df['acq_date'], format='%Y-%m-%d').dt.to_period('Q').astype(str)
                name = name.replace("csv", "parquet.gzip")
                df_list.append(df)

    df = pd.concat(df_list)

    return df

def aggregate(fromdate,todate,priogrid, popualtion,preprocessed,rootdirsearch):

    if not os.path.exists(f"{preprocessed}/fire_count_pgid.parquet.gzip"):

        df = process_data(rootdirsearch,priogrid, popualtion)
        df.to_parquet(f"{preprocessed}/fire_count_pgid.parquet.gzip", compression="gzip")
    else:
        df = pd.read_parquet(f"{preprocessed}/fire_count_pgid.parquet.gzip")

    df = df.loc[((df.quarter>=fromdate) & (df.quarter<=todate))]
    df=df.groupby(["pgid", "lat", "lon"]).sum().reset_index()[["pgid", "lat", "lon","count_fire"]]

    df = df.merge(popualtion[["pgid","wp_pop_density"]],on="pgid",how="left")

    df = custom_norm(df, "count_fire")

    normality_test(df[["count_fire_multiply_pop_density_log_minmax"]])

    df.to_csv(f"fire_{fromdate}-{todate}.csv")

    return df



if __name__ == '__main__':


    preprocessed="D:/DATA/FIRMS/pgid_preprocessed"
    rootdirsearch = "D:/DATA/FIRMS/MODIS/modis/"

    priogrid = pd.read_parquet(
        "C:/Users/email/Documents/conflictproxyindicators/reference_datasets/base-grid/base_grid_prio.parquet").reset_index()

    popualtion = pd.read_parquet(
        "../../../reference_datasets/population/population_worldpop.parquet").reset_index()

    popualtion = popualtion.loc[((popualtion.year == 2023) & (popualtion.quarter == 4))]

    df = aggregate("2022Q3", "2023Q3", priogrid, popualtion,preprocessed,rootdirsearch)

    df = df[["pgid","boxcoxb_log_minmax"]]
    df.columns = ["pgid","CLI_risk_fires_12m"]
    df["year"] = 2023
    df["quarter"] = 3
    df.to_parquet("CLI_risk_fires_12m.parquet")

    df = aggregate("2017Q3", "2023Q3", priogrid, popualtion,preprocessed,rootdirsearch)

    df = df[["pgid","boxcoxb_log_minmax"]]
    df.columns = ["pgid","CLI_risk_fires_7y"]
    df["year"] = 2023
    df["quarter"] = 3
    df.to_parquet("CLI_risk_fires_7y.parquet")

else:
    import sys


    from_q = sys.argv[1]
    to_q = sys.argv[2]
    out_filename=sys.argv[3]


    preprocessed="/DATA/REFERENCE_DATASETS/FIRMS/pgid_preprocessed"
    rootdirsearch = "/DATA/REFERENCE_DATASETS/FIRMS/MODIS/modis/"

    priogrid = pd.read_parquet(
        "/DATA/REFERENCE_DATASETS/BASEGRID/base_grid_prio.parquet").reset_index()

    popualtion = pd.read_parquet(
        "/DATA/REFERENCE_DATASETS/POPULATION/population_worldpop.parquet").reset_index()

    popualtion = popualtion.loc[((popualtion.year == 2023) & (popualtion.quarter == 4))]

    df = aggregate("from_q", "to_q", priogrid, popualtion,preprocessed,rootdirsearch)

    df = df[["pgid","boxcoxb_log_minmax"]]
    df.columns = ["pgid",out_filename.replace("parquet","")]
    df["year"] = 2023
    df["quarter"] = 3
    df.to_parquet(out_filename)



