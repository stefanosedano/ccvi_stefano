import tropycal.tracks as tracks
import datetime as dt
import ssl
from cyclons_lib import gridded_stats_
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import os
import sys
sys.path.append('..')
from general_functions_ccvi.normality_test import normality_test
from general_functions_ccvi.log_with_pop_and_fit_normal_distribution import custom_norm
import warnings
warnings.filterwarnings('ignore')

ssl._create_default_https_context = ssl._create_unverified_context

def process_points(points,datefrom,dateto):
    points["count_cyclones"]=1
    points_grouped = points.loc[((points.quarter >= datefrom) & (points.quarter <= dateto))].groupby(["latbin", "lonbin"]).sum().reset_index()
    points_grouped = points_grouped[['latbin', 'lonbin', 'count_cyclones']]
    points_grouped.columns = ['latbin', 'lonbin', 'count_cyclones']
    return points_grouped


def cyclons_to_grid(binsize=0.5,datefrom=None,dateto=None,ibtrak_url=""):
    import requests
    temp_csv = "temporary_cyclons_tracks.csv"
    if not os.path.exists(temp_csv):
        url = ibtrak_url
        response = requests.get(url)
        open(temp_csv, "wb").write(response.content)

    ibtracs = tracks.TrackDataset(basin='all',source='ibtracs',ibtracs_mode='jtwc_neumann',catarina=True,ibtracs_url=temp_csv)
    points = gridded_stats_(ibtracs,request="maximum wind",year_range=(datefrom.year,dateto.year),  return_array=True, binsize=binsize)

    return points


def aggregate(fromdate,todate,priogrid,popualtion,ibtrak_url,indicator_name):



    filename_full = f"cyclons_{fromdate}-{todate}_full.csv"
    filename = f"cyclons_{fromdate}-{todate}.csv"
    if not os.path.exists("cyclones_point.parquet.gz"):
        print("building cyclons point df")
        points = cyclons_to_grid(binsize=1, datefrom=pd.to_datetime(fromdate), dateto=pd.to_datetime(todate),ibtrak_url=ibtrak_url)
        points.to_parquet("cyclones_point.parquet.gz", compression="gzip")
    else:
        print("reading cyclons point df from cache")
        points = pd.read_parquet("cyclones_point.parquet.gz")


    print("--> mapping to bins")
    points['quarter'] = pd.to_datetime(points['time']).dt.to_period('Q').astype(str)
    binsize=1
    def to_bin(x):
        if x > 180:
            x = -180 + x -180
        return (binsize / 2) + np.floor(x / binsize) * binsize

    points["latbin"] = points.lat.map(to_bin)
    points["lonbin"] = points.lon.map(to_bin)

    popualtion["latbin"] = popualtion.lat.map(to_bin)
    popualtion["lonbin"] = popualtion.lon.map(to_bin)


    df = process_points(points,fromdate,todate)

    df.to_csv(filename_full, index=False)

    df.to_parquet(f"raw_{indicator_name}.parquet")

    out=pd.merge(popualtion,df, on=["latbin","lonbin"], how="left")
    out=out.loc[~(out["count_cyclones"].isna())]

    out = custom_norm(out, "count_cyclones")

    normality_test(out[["boxcoxb_log"]])

    out.to_csv(f"cyclons_{fromdate}-{todate}.csv")

    return out


if __name__ == '__main__':

    import sys
    print(f"reference quarter {sys.argv[1]}")


    date_from = sys.argv[1]
    date_to = sys.argv[2]
    year = date_to[:4]
    quarter = date_to[-1]

    outputdir = sys.argv[3]

    indicator_name = sys.argv[4].replace(".parquet","")

    ibtrak_url = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.ALL.list.v04r00.csv"
    priogrid = pd.read_parquet(
        "/DATA/REFERENCE_DATASETS/BASEGRID/base_grid_prio.parquet").reset_index()

    popualtion = pd.read_parquet(
        "/DATA/REFERENCE_DATASETS/POPULATION/population_worldpop.parquet").reset_index()

    popualtion = popualtion.loc[((popualtion.year == 2023) & (popualtion.quarter == 4))]


    df = aggregate(date_from, date_to,priogrid,popualtion,ibtrak_url,indicator_name)
    df = df[["pgid","boxcoxb_log_minmax"]]
    df.columns = ["pgid",indicator_name]
    df["year"] = year
    df["quarter"] = quarter
    df.to_parquet(f"{outputdir}/{indicator_name}.parquet")
