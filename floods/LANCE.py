import pandas as pd
import requests
from bs4 import BeautifulSoup
import regex as re
import tarfile
import urllib
from requests.auth import HTTPBasicAuth
import glob
import gzip
from io import BytesIO
import rasterio
import rasterio.plot
import matplotlib
import matplotlib.pyplot as plt
import os




token = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6InN0ZWZhbm9mZXJyaTE5NzgiLCJleHAiOjE3MDExODcxMDQsImlhdCI6MTY5NjAwMzEwNCwiaXNzIjoiRWFydGhkYXRhIExvZ2luIn0.prmQjFwLt0EBZjU-2fvFbLxYmcmhsDd5D1aiuqQ4iAYd7I2QO0gUftzr3SdVjRiCuUtp_IZ9UjmVudx78hwpyVvY19w4ElTB-ZGwEOY5D3UAqoW6SHStxIsDLf4ALkNhsn35_HJv3fkkJzif0n_SDZ_OEyReKdwV87il9P6C4I6S2Mp4kHX7Y9Li2ueBSu0wJAEbkB-_R5QFKvWNltTJ5PCLZL_OXnCEpVwq19lW09RiERe232mIFCVomTXSGI385B-As-Ve7iz9XV6fWjHUDThEuOkL42KpjpxfwV5q9BmoIhaTEz-Trvb_Mp8ZjisEj9b9B2a-QmcNLKV1PqleGw"


viirs_path = "https://eogdata.mines.edu/wwwdata/dmsp/monthly_composites/by_year/"




def gen_fullpath():

    Lista_full_path=[]
    for year in range(2000,2009):
        Lista_full_path.append(f"{viirs_path}/{str(year)}/")

    return Lista_full_path


def path_dawnload():
    url = gen_fullpath()
    Lista_path = []
    for x in url:
        page = requests.get(x)
        soup = BeautifulSoup(page.text, 'html.parser')
        if page.status_code == 200:
            links = soup.find_all(href=True)
            for link in links:
                link = link['href']
                if "vis.tif" in link:
                    path = x + link
                    Lista_path.append(path)
        else:
            print("No se encontro la pagina web")
    return Lista_path

Lista_path=path_dawnload()

import sys
from requests.auth import HTTPBasicAuth
from pathlib import Path
from tqdm.auto import tqdm


def runDownload():

    import requests
    import json
    import os
    user = "email4stefanoferri@gmail.com"
    password = "mRUW99QM83Zqd.a"
    # Retrieve access token
    params = {
        'client_id': 'eogdata_oidc',
        'client_secret': '2677ad81-521b-4869-8480-6d05b9e57d48',
        'username': user,
        'password': password,
        'grant_type': 'password'
    }
    token_url = 'https://eogauth.mines.edu/auth/realms/master/protocol/openid-connect/token'
    response = requests.post(token_url, data=params)
    access_token_dict = json.loads(response.text)
    access_token = access_token_dict.get('access_token')
    # Submit request with token bearer
    ## Change data_url variable to the file you want to download

    for liks in Lista_path:

        x = liks.split("/")
        pathtgz = x[-1]
        pathtgz = pathtgz.split(".")
        pathtgz =pathtgz[0]

        data_url = liks

        output_file = f"D:/DATA/NIGHTLIGHT/monthlycomposite/{pathtgz[4:8]}/{pathtgz}.tif"
        if not os.path.exists(output_file):

            auth = 'Bearer ' + access_token
            headers = {'Authorization': auth}
            response = requests.get(data_url, headers=headers)
            # Write response to output file
            ## You can either define the output file name directly
            # output_file = 'EOG_sensitive_contents.txt'
            ## Or get the filename from the data_url variable
            if not os.path.exists(f"D:/DATA/NIGHTLIGHT/monthlycomposite/{pathtgz[4:8]}"):
                os.makedirs(f"D:/DATA/NIGHTLIGHT/monthlycomposite/{pathtgz[4:8]}")

            output_file = f"D:/DATA/NIGHTLIGHT/monthlycomposite/{pathtgz[4:8]}/{pathtgz}.tif"
            with open(output_file, 'wb') as f:
                f.write(response.content)


runDownload()