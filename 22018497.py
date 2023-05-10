# -*- coding: utf-8 -*-
"""
Created on Tue May  9 22:14:14 2023

@author: aaron
"""

import pandas as pd
from sklearn.cluster import KMeans
import cluster_tools as ct
import matplotlib.pyplot as plt


def read_df(fname):
    df = pd.read_csv(fname, skiprows=4)
    df.drop(["Country Code", "Indicator Code", "Indicator Name"], axis=1,
            inplace=True)
    df.set_index(['Country Name'], inplace=True)
    df.index.rename("Country", inplace=True)
    df.dropna(how="all", axis=1, inplace=True)
    df.dropna(how="any", thresh=50, inplace=True)
    df.fillna(0, inplace=True)
    # transposing
    df_t = df.transpose()

    # droping columns with nan values
    df_t.dropna(how='all', axis=1, inplace=True)

    return df, df_t


df_ogc, df_ogc_t = read_df("API_EG.ELC.FOSL.ZS_DS2_en_csv_v2_5211704.csv")

years = ['1960', '1961', '1962', '1963', '1964', '1965', '2010', '2011',
         '2012', '2013', '2014', '2015']
pd.plotting.scatter_matrix(df_ogc[years], figsize=(12, 12))
plt.show()
