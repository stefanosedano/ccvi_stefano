import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import shapiro
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, MinMaxScaler, Normalizer
sys.path.append('..')
from general_functions_ccvi.normality_test import normality_test
from general_functions_ccvi.log_with_pop_and_fit_normal_distribution import custom_norm
import warnings
warnings.filterwarnings('ignore')

def aggregate(fromdate,todate, priogrid, popualtion,indicator_name):

    archive = "https://floodobservatory.colorado.edu/temp/FloodArchive.xlsx"
    recent_events="https://floodobservatory.colorado.edu/Version3/MasterListrev.htm"


    df_archive=pd.read_excel(archive)[["ID","long","lat","Began","Ended","Severity"]]
    list_of_archive_ids=df_archive[["ID"]]
    df_recent_events=pd.read_html(recent_events,header=0)[0][["Register #","Centroid X","Centroid Y","Began","Ended","Severity *"]]
    df_recent_events = df_recent_events.loc[~df_recent_events["Register #"].isin(list_of_archive_ids)]


    df_recent_events.columns=["ID","long","lat","Began","Ended","Severity"]

    df = pd.concat([df_recent_events,df_archive])

    df = df.loc[~df.long.isna()]

    df['Began'] = pd.to_datetime(df['Began'])
    df['Ended'] = pd.to_datetime(df['Ended'])
    df['num_days'] = (df['Ended'] - df['Began']).dt.days
    df['YEAR'] = df['Ended'].dt.year

    print("--> mapping to bins")
    binsize=1
    def to_bin(x):
        if x > 180:
            x = -180 + x -180
        return (binsize / 2) + np.floor(x / binsize) * binsize

    df["latbin"] = df.lat.map(to_bin)
    df["lonbin"] = df.long.map(to_bin)

    df=df[['ID', 'latbin', 'lonbin', 'Began', 'Ended', 'Severity', 'num_days', 'YEAR']]
    df.columns=['ID', 'latbin', 'lonbin', 'Began', 'Ended', 'Severity', 'num_days', 'YEAR']

    df["quarter"] = pd.to_datetime(df['Began']).dt.to_period('Q').astype(str)
    grouped_df = df.loc[((df["quarter"] >= fromdate) & (df["quarter"] <= todate))].groupby(["latbin", "lonbin"]).sum().reset_index()
    grouped_df.to_csv(f"flood_{fromdate}-{todate}_full.csv")

    grouped_df.to_parquet(f"raw_{indicator_name}.parquet")


    popualtion["latbin"] = popualtion.lat.map(to_bin)
    popualtion["lonbin"] = popualtion.lon.map(to_bin)


    grouped_df = pd.merge(popualtion, grouped_df, on=["latbin", "lonbin"], how="left").reset_index()
    grouped_df = grouped_df.loc[~grouped_df.num_days.isna()]

    out = custom_norm(grouped_df, "num_days")

    normality_test(out[["boxcoxb_log_minmax"]])

    out.to_csv(f"flood_{fromdate}-{todate}.csv")

    return out


if __name__ == '__main__':


    import sys
    #print("define the args:")
    #print("from quarter eg 2022Q1")
    #print("from to quarter eg 2023Q4")
    #print("outfilename eg CLI_risk_fires_7y.parquet")

    from_q = sys.argv[1]
    to_q = sys.argv[2]
    out_dir =sys.argv[3]
    out_filename=sys.argv[4]

    priogrid = pd.read_parquet(
        "/DATA/REFERENCE_DATASETS/BASEGRID/base_grid_prio.parquet").reset_index()

    popualtion = pd.read_parquet(
        "/DATA/REFERENCE_DATASETS/POPULATION/population_worldpop.parquet").reset_index()

    popualtion = popualtion.loc[((popualtion.year == 2023) & (popualtion.quarter == 4))]

    indicator_name = out_filename.replace(".parquet","")
    df = aggregate(from_q, to_q, priogrid, popualtion, indicator_name)


    df = df[["pgid","boxcoxb_log_minmax"]]
    df.columns = ["pgid",indicator_name]
    df["year"] = from_q[:4]
    df["quarter"] = from_q[-1]
    df.to_parquet(f"{out_dir}/{out_filename}")

