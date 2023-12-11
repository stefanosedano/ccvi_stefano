#https://iopscience.iop.org/article/10.1088/1748-9326/10/12/124003

import os
import ee
import numpy as np
import pandas as pd
from functools import reduce
import warnings
warnings.filterwarnings('ignore')

warnings.simplefilter(action='ignore', category=FutureWarning)
import time
from datetime import datetime, timedelta

def build_percentile_per_pgid(arr):
    out=[]
    for i in range(arr.shape[0]):
        if (arr[i,0,:].max() == arr[i,0,:].min()):
            pgid = int(arr[i,0,0])
            percentile_distribution_95 = np.percentile(arr[i,1,:],95 )
            out.append([pgid,percentile_distribution_95])
        else:
            print("some errors in the pgid")
    out = pd.DataFrame(out,columns=["pgid","95p"])
    return out

def get_ranges(centerdate):
    r = pd.date_range((centerdate - timedelta(days=15)), (centerdate + timedelta(days=15)), freq='d')
    list_of_month_and_days = []
    for el in r:
        list_of_month_and_days.append([el.day,el.month])

    list_of_month_and_days = pd.DataFrame(list_of_month_and_days,columns=["day","month"])
    list_of_month_and_days = list_of_month_and_days.drop_duplicates()
    list_of_layers = []
    for year in range(1951,2011):
        for el in list_of_month_and_days.values:
            try:
                file = f"/DATA/REFERENCE_DATASETS/ERA5/precipitation/raw/era5_precipitation/{year}/image_{year}-{str(el[1]).zfill(2)}-{str(el[0]).zfill(2)}.parquet.gzip"
                df = pd.read_parquet(file)[["pgid","sum"]].sort_values(by=['pgid']).to_numpy()

                list_of_layers.append(df)
            except Exception as e:
                print(e)
                pass


    list_of_layers = np.dstack(list_of_layers)
    df = build_percentile_per_pgid(list_of_layers)
    return df






if __name__ == '__main__':
    #366 treshold considering leap year years
    ##1991-2020 reference epriod!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    startDate = datetime(2024, 1, 1)
    endDate = datetime(2024, 12, 31)
    renge_dates = pd.date_range(startDate, endDate, freq='d')
    reference_directory = "/DATA/REFERENCE_DATASETS/ERA5/precipitation/raw/era5_precipitation_reference"
    for day in renge_dates:
        if not os.path.exists(f"{reference_directory}/{str(day.month).zfill(2)}_{str(day.day).zfill(2)}.parquet.gzip"):
            try:
                reference = get_ranges(day)

                reference.to_parquet(f"{reference_directory}/{str(day.month).zfill(2)}_{str(day.day).zfill(2)}.parquet.gzip", compression="gzip")
            except Exception as e:
                print(e)
                pass









