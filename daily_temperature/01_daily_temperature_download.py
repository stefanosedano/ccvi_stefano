import os
import ee
import numpy as np
import pandas as pd
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')

def fc_to_dict(fc):
    prop_names = fc.first().propertyNames()
    prop_lists = fc.reduceColumns(
        reducer=ee.Reducer.toList().repeat(prop_names.size()), selectors=prop_names
    ).get("list")

    return ee.Dictionary.fromLists(prop_names, prop_lists)

def pad_dict_list(dict_list, padel):
    lmax = 0
    for lname in dict_list.keys():
        lmax = max(lmax, len(dict_list[lname]))
    for lname in dict_list.keys():
        ll = len(dict_list[lname])
        if ll < lmax:
            dict_list[lname] += [padel] * (lmax - ll)
    return dict_list

def f(x):

    startDate= x["startdate"]
    temp_files = x["temp_files"]

    if not os.path.exists(f"{temp_files}/raw/era5_temperature_max/{startDate[:4]}"):
        os.makedirs(f"{temp_files}/raw/era5_temperature_max/{startDate[:4]}")

    outfile = f"{temp_files}/raw/era5_temperature_max/{startDate[:4]}/image_{startDate}.parquet.gzip"

    doneIt = False
    if not os.path.exists(outfile):
        print(f"downlading: {outfile}")

        credentials = ee.ServiceAccountCredentials(
            "test1-landsat8@geelandsat8.iam.gserviceaccount.com",
            "../geelandsat8-6af86334d7ec.json",
        )
        ee.Initialize(credentials)
        my_grid = ee.FeatureCollection("projects/geelandsat8/assets/base_grid_prio")

        #while not doneIt:
        try:
            myPanelData = ZsGEE()
            myPanelData.startDate = startDate
            myPanelData.satellite = "ECMWF/ERA5_LAND/DAILY_AGGR"
            myPanelData.bands = "temperature_2m_max"
            myPanelData.asset = my_grid
            df = myPanelData.get_dataframe()
            doneIt = True

            df.to_parquet(outfile, compression="GZIP")

        except Exception as e:
            print(f"date: {startDate}, error: {e}")
            pass
    else:
        print(f"Exists in cache!!: {outfile}")


class ZsGEE:

    def __init__(self,):

        # some init here
        self.some_var_here = ""

    def get_dataframe(self):

        startDate =  self.startDate  # '2000-03-01' # time period of interest beginning date
        interval = 1  # time window length
        intervalUnit = "day" #'month'  # unit of time e.g. 'year', 'month', 'day'
        intervalCount = 1  # 275 # number of time windows in the series
        dataset = ee.ImageCollection(self.satellite).select(self.bands)
        temporalReducer = ee.Reducer.mean()  # how to reduce images in time window
        spatialReducers = ee.Reducer.mean()  # how to reduce images in time window
        # Get time window index sequence.
        intervals = ee.List.sequence(0, intervalCount - 1, interval)

        # Map reductions over index sequence to calculate statistics for each interval.
        def a(i):
            # Calculate temporal composite.
            startRangeL = ee.Date(startDate).advance(i, intervalUnit)
            endRangeL = startRangeL.advance(interval, intervalUnit)
            temporalStat = dataset.filterDate(startRangeL, endRangeL).reduce(temporalReducer)

            # Calculate zonal statistics.
            statsL = temporalStat.reduceRegions(
                collection=self.asset,
                reducer=spatialReducers,
                scale= dataset.first().projection().nominalScale().getInfo(),
                crs=dataset.first().projection()
            )

            # Set start date as a feature property.

            def b(feature):
                #  or 'YYYY-MM-dd'
                return feature.set({'composite_start': startRangeL.format('YYYYMMdd')})

            return statsL.map(b)

        zonalStatsL = intervals.map(a)

        zonalStatsL = ee.FeatureCollection(zonalStatsL).flatten()

        output = fc_to_dict(zonalStatsL).getInfo()
        output = pad_dict_list(output, np.nan)
        out_pd = pd.DataFrame(output)
        out_pd['quarter'] = pd.to_datetime(out_pd['composite_start'], format='%Y%m%d').dt.to_period('Q').astype(str)
        out_pd['q'] = out_pd['quarter'].str.strip().str[-2:]
        if not os.path.exists("/DATA/ERA5/tempertaure/latlon.parquet.gzip"):
            out_pd[["pgid","lat","lon"]].to_parquet("latlon.parquet.gzip",compression="gzip")
        return out_pd[['pgid','composite_start', 'mean','quarter',"q"]]



if __name__ == '__main__':

    import time
    from datetime import datetime, timedelta
    import sys

    year = int(sys.argv[1])
    month = int(sys.argv[2])
    day = int(sys.argv[3])

    start = time.time()
    temp_files = "/DATA/REFERENCE_DATASETS/ERA5/tempertaure/"

    if not os.path.exists(temp_files):
        os.makedirs(temp_files)
    from_year = 1951
    to_year = year
    cut_off_year = 2011
    ####DOWNLOAD DATA FROM GEE

    params = []

    startDate = datetime(1951, 1, 1)
    endDate = datetime(year, month, day)

    # Getting List of Days using pandas
    datesRange = pd.date_range(startDate, endDate - timedelta(days=1), freq='d')


    for datei in datesRange:
        parx = {
            "startdate": datei.strftime("%Y-%m-%d"),
            "temp_files": temp_files
        }
        params.append(parx)

    with Pool(30) as p:
        p.map(f, params)




