#archive:
#https://firms.modaps.eosdis.nasa.gov/download/
import time
from datetime import datetime, timedelta
import requests
import pandas as pd
import zipfile
import os
import json
import schedule
import time
import numpy as np
import io

allJobsDone=False


##download archive first (2000-2022)
def download_archive(fromyear,toyear):
    for year in range(fromyear, toyear):
        outputfile = f"/DATA/REFERENCE_DATASETS/FIRMS/MODIS/modis_{year}_all_countries.zip"
        if not os.path.exists(outputfile):
            url = f"https://firms.modaps.eosdis.nasa.gov/data/country/zips/modis_{year}_all_countries.zip"
            response = requests.get(url)

            open(outputfile, "wb").write(response.content)
            with zipfile.ZipFile(outputfile, "r") as zip_ref:
                zip_ref.extractall(f"D:/DATA/FIRMS/MODIS")



def download_latest(year):
    #get last year world data (2023)
    def get_starting_dates(start_date, end_date, interval_days):
        starting_dates = []
        current_date = start_date

        while current_date <= end_date:
            starting_dates.append(current_date)
            current_date += timedelta(days=interval_days)

        return starting_dates

    def job():
        # get staritng date for NRL modis
        url = "https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/61d3c440fba85794109ad61bbe0cc221/MODIS_NRT"
        urlData = requests.get(url).content
        rawData = pd.read_csv(io.StringIO(urlData.decode('utf-8')))
        fromdate = pd.to_datetime(rawData["min_date"].values[0])

        checkapikey = "https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY=61d3c440fba85794109ad61bbe0cc221"
        response = requests.get(checkapikey)
        staus = response.content
        status = json.loads(staus)
        if status["current_transactions"] < 1000:
            start_date = datetime(year, 1, 1)  # Specify your start date (year, month, day)
            end_date = datetime(year, 12, 31)  # Specify your end date (year, month, day)
            interval_days = 1  # Specify the interval in days

            starting_dates = get_starting_dates(start_date, end_date, interval_days)

            checkapikey = "https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY=61d3c440fba85794109ad61bbe0cc221"

            for datei in starting_dates:

                if (datei >= fromdate):
                    model = "MODIS_NRT"
                else:
                    model = "MODIS_SP"
                date_to_download = datei.strftime("%Y-%m-%d"),
                outputfile = f"/DATA/REFERENCE_DATASETS/FIRMS/MODIS/modis/{year}/{date_to_download[0]}.csv"
                if not os.path.exists(outputfile):
                    urlquery = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/61d3c440fba85794109ad61bbe0cc221/{model}/-180,-90,180,90/1/{date_to_download[0]}"
                    response = requests.get(urlquery)
                    if "Exceeding allowed transaction limit" in str(response.content):
                        break
                    outputfile = f"/DATA/REFERENCE_DATASETS/FIRMS/MODIS/modis/{year}/{date_to_download[0]}.csv"
                    open(outputfile, "wb").write(response.content)

            global allJobsDone
            allJobsDone = True



    # Schedule the job to run every minute
    schedule.every(1).minutes.do(job)

    # Run the scheduled tasks


    while (not allJobsDone):
        schedule.run_pending()
        time.sleep(1)  # Wait for 1 second before checking again

if __name__ == '__main__':
    import sys

    #print(f"reference quarter {sys.argv[1]}")

    year = int(sys.argv[1][:4])

    from_date = 2000
    last_date = year
    download_archive(from_date, last_date)
    download_latest(last_date)
