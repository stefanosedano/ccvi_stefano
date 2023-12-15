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

#apply the formula from the paper
def calculate_magnitude(heatwave,arr_temp,t_25,t_75):
    list_of_magnitudes=[]
    for id in heatwave:
        if arr_temp[id] >t_25[id]:
            list_of_magnitudes.append((arr_temp[id] - t_25[id])/(t_75[id]-t_25[id]))
        else:
            list_of_magnitudes.append(0)

    #HWMId magnitude: sum of the magnitude of the consecutive days composing a heatwave
    return np.array(list_of_magnitudes).sum()

def calculate_magnitude_precipitation(heatwave,arr_temp):
    list_of_magnitudes=[]
    for id in heatwave:
        list_of_magnitudes.append(arr_temp[id])

    #HWMId magnitude: sum of the magnitude of the consecutive days composing a heatwave
    #add amount of water
    return np.array(list_of_magnitudes).sum()


def count_anomaly_and_magnitude(arr_temp,pgid,renge_dates,reference_pgid):

    t_95 = []

    #building years vectors: (365/366 days) of the 90h, 75h, 25h percentile for the specific pgid
    for i in range(0,len(arr_temp)):
        day = renge_dates[i]
        dataref = f"{str(day.month).zfill(2)}_{str(day.day).zfill(2)}"
        reference = reference_pgid.loc[reference_pgid.dateref==dataref]

        t_95.append(reference["95p"].values[0])

    #vectors series of referece values
    t_95 = np.array(t_95)

    # Heatwave: period of 3 consecutive days with maximum temperature (Tmax) above the daily threshold
    # for the reference period 1981–2010. The threshold is
    # defined as the 90th percentile of daily maxima temperature, centered on a 31 day window
    anomaly = arr_temp > t_95
    anomaly = anomaly * 1

    # recording the position (day) of the anomaly when are >=3 cosecutive days
    heatwave_list = []
    heatwave = []

    for i in range(0, anomaly.shape[0]):
        if anomaly[i] == 1:
            heatwave.append(i)
        elif anomaly[i] == 0:
            if len(heatwave) >= 1:
                heatwave_list.append(heatwave)
                heatwave = []
            else:
                heatwave = []

    number_of_heatwaves = len(heatwave_list)

    magnitude_heatwaves = []
    #for heatwave in heatwave_list:
    #    magnitude_heatwaves.append(calculate_magnitude_precipitation(heatwave, arr_temp))

    # It is defined as the maximum magnitude of the heatwaves in a year
    if len(magnitude_heatwaves) > 0:
        magnitude = np.array(magnitude_heatwaves).max()
    else:
        magnitude = 0

    return number_of_heatwaves, magnitude





def build_anomaly(arr,renge_dates,reference_pd):
    out=[]
    #loop for all pgid to get and scan the temporal series

    for i in range(arr.shape[0]):
        print(f"processing pgid {i} on {arr.shape[0]}")
        if (arr[i,0,:].max() == arr[i,0,:].min()):

            #identify the pgid
            pgid = int(arr[i, 0, 0])

            # identify the 366 reference values 90h, 75h, 25h for the selected pgid
            reference_pgid=reference_pd.loc[reference_pd.pgid==pgid]

            #arr[i,1,:] is the series of tmax value for the selected pgid (356/366 values)
            #pgid is pgid
            #reference_pgid is a DataFrame of all days reference values for the selected pgid
            count_an,magnitude_an = count_anomaly_and_magnitude(arr[i,1,:],pgid,renge_dates,reference_pgid)

            out.append([pgid,count_an,magnitude_an])

        else:
            print("some errors in the pgid")
    out = pd.DataFrame(out,columns=["pgid","count","magnitude"])

    return out




def aggregate(fromdate, todate,popualtion,priogrid,path_daily_prec,path_daily_prec_reference,path_year_anomaly,indicator_name):

    import sys
    #path_daily_temp = "D:/DATA/ERA5/temp/precipitation/raw/era5_precipitation"
    #path_daily_temp_reference = "D:/DATA/ERA5/temp/precipitation/raw/era5_precipitation_reference/"
    #path_year_anomaly = "D:/DATA/ERA5/temp/precipitation/raw/anomaly/"

    # last year measure, most recent quarter 2023Q2
    yearquarterto = todate  # (sys.argv[1])
    yearquarterfrom = fromdate  # (sys.argv[1])

    startDate = datetime(1951, 1, 1)
    endDate = datetime(2024, 8, 3)

    if not os.path.exists(f"{path_year_anomaly}/{yearquarterfrom}-{yearquarterto}_year.csv"):

        print(f"building cache anomaly: {path_year_anomaly}/{yearquarterfrom}-{yearquarterto}_year.csv")

        renge_dates = pd.date_range(startDate, endDate, freq='d')

        renge_dates = pd.DataFrame(renge_dates, columns=["date"])
        renge_dates['quarter'] = pd.to_datetime(renge_dates['date'], format='%Y%m%d').dt.to_period('Q').astype(str)
        renge_dates['q'] = renge_dates['quarter'].str.strip().str[-2:]
        renge_dates_max = renge_dates.loc[renge_dates.quarter == yearquarterto]
        renge_dates_min = renge_dates.loc[renge_dates.quarter == yearquarterfrom]
        # renge_dates = renge_dates.date.to_list()

        from_date = renge_dates_min.date.min()
        to_date = renge_dates_max.date.max()
        renge_dates = renge_dates.loc[(renge_dates.date >= from_date) & (renge_dates.date <= to_date)]
        renge_dates = renge_dates.date.to_list()

        # Grouping all the 366 reference DataFrame that contains per pgid the Tmax90,Tmax75,Tmax25
        reference_pd = []
        for filename in os.listdir(path_daily_prec_reference):
            f = os.path.join(path_daily_prec_reference, filename)
            df = pd.read_parquet(f)
            month = filename[0:2]
            day = filename[3:5]
            df["dateref"] = f"{month}_{day}"
            reference_pd.append(df)

        reference_pd = pd.concat(reference_pd)

        # Grouping the daily Tmax in the selected year
        df = []
        for day in renge_dates:
            data = f"{path_daily_prec}/{day.year}/image_{day.year}-{str(day.month).zfill(2)}-{str(day.day).zfill(2)}.parquet.gzip"
            dfi = pd.read_parquet(data)[["pgid", "sum"]].sort_values(by=['pgid']).to_numpy()
            df.append(dfi)

        df = np.dstack(df)

        df_anomaly = build_anomaly(df, renge_dates, reference_pd)
        latlon = pd.read_parquet("../latlon.parquet.gzip")
        df_anomaly = df_anomaly.merge(latlon, on="pgid", how="left")
        df_anomaly.to_parquet(f"{path_year_anomaly}/{yearquarterfrom}-{yearquarterto}_year.parquet.gzip")
        df_anomaly.to_csv(f"{path_year_anomaly}/{yearquarterfrom}-{yearquarterto}_year.csv")


    df_anomaly = pd.read_parquet(f"{path_year_anomaly}/{yearquarterfrom}-{yearquarterto}_year.parquet.gzip")


    df_anomaly=df_anomaly[["pgid","count"]]

    df_anomaly.to_parquet(f"raw_{indicator_name}.parquet")

    df_anomaly=df_anomaly.merge(popualtion, on="pgid", how="left")

    sys.path.append('..')
    from general_functions_ccvi.normality_test import normality_test
    from general_functions_ccvi.log_with_pop_and_fit_normal_distribution import custom_norm

    out = custom_norm(df_anomaly, "count")

    normality_test(out[["boxcoxb_log_minmax"]])

    out.to_csv(f"precipitations_anomaly_{yearquarterfrom}-{yearquarterto}.csv")

    return out


if __name__ == '__main__':


    path_daily_temp = "/DATA/REFERENCE_DATASETS/ERA5/precipitation/raw/era5_precipitation_max"
    path_daily_temp_reference = "/DATA/REFERENCE_DATASETS/ERA5/precipitation/raw/era5_precipitation_max_reference/"
    path_year_anomaly = "/DATA/REFERENCE_DATASETS/ERA5/precipitation/raw/anomaly/"

    if not os.path.exists(path_daily_temp):
        os.makedirs(path_daily_temp)

    if not os.path.exists(path_daily_temp_reference):
        os.makedirs(path_daily_temp_reference)

    if not os.path.exists(path_year_anomaly):
        os.makedirs(path_year_anomaly)

    path_population="/DATA/REFERENCE_DATASETS/POPULATION/population_worldpop.parquet"
    path_priogrid= "/DATA/REFERENCE_DATASETS/BASEGRID/base_grid_prio.parquet"

    population = pd.read_parquet(path_population).reset_index()
    population = population.loc[((population.year == 2023) & (population.quarter == 4))]
    priogrid =  pd.read_parquet(path_priogrid).reset_index()

    import sys


    from_q = sys.argv[1]
    to_q = sys.argv[2]
    out_dir = sys.argv[3]
    out_filename = sys.argv[4]

    indicator_name = out_filename.replace(".parquet","")

    df = aggregate(from_q, to_q,population,priogrid,path_daily_temp,path_daily_temp_reference,path_year_anomaly,indicator_name)
    df = df[["pgid","boxcoxb_log_minmax"]]
    df.columns = ["pgid",indicator_name]
    df["year"] = to_q[:4]
    df["quarter"] = to_q[-1]
    df.to_parquet(f"{out_dir}/{indicator_name}.parquet")
