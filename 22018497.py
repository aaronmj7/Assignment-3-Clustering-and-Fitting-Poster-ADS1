# -*- coding: utf-8 -*-
"""
Created on Tue May  9 22:14:14 2023

@author: aaron
"""

import pandas as pd
from sklearn.cluster import KMeans
import cluster_tools as ct
import matplotlib.pyplot as plt
import seaborn as sns


def read_df(fname):
    # reading from csv
    df = pd.read_csv(fname, skiprows=4)

    # dropping columns that are not needed
    df.drop(["Country Code", "Indicator Code", "Indicator Name"], axis=1,
            inplace=True)
    # setting index
    df.set_index(['Country Name'], inplace=True)
    # droping columns and rows with nan values
    df.dropna(how="all", axis=1, inplace=True)
    df.dropna(how="any", thresh=50, inplace=True)
    # filling nan values
    df.fillna(0, inplace=True)

    # transposing
    df_t = df.transpose()
    # droping columns with nan values
    df_t.dropna(how='all', axis=1, inplace=True)

    return df, df_t


df_ogc, df_ogc_t = read_df("API_EG.ELC.FOSL.ZS_DS2_en_csv_v2_5211704.csv")

# setting seaborn theme
sns.set_theme()

# plotting scatter matrix to find good cluster
years = ['1960', '1961', '1962', '1963', '1964', '1965',
         '2010', '2011', '2012', '2013', '2014', '2015']
pd.plotting.scatter_matrix(df_ogc[years], figsize=(12, 12))
plt.show()

# chosing best years for clustering
df_clstr = df_ogc[['1965', '2015']].copy()

# normalising and storing minimum and maximum
df_normalised, df_min, df_max = ct.scaler(df_clstr)

# calculating wcss for different no. of clusters
wcss = []
for i in range(1, 10):
    # set up kmeans and fit
    kmeans = KMeans(n_clusters=i, max_iter=300, n_init=10)
    kmeans.fit(df_normalised)
    wcss.append(kmeans.inertia_)

# plotting to use elbow method
plt.figure(figsize=(14, 7))

plt.plot(range(1, 10), wcss, marker='o')

plt.title('The Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Plot for two clusters

nc = 2
kmeans = KMeans(n_clusters=nc)
# fit data
kmeans.fit(df_normalised)
# extracting labels
labels = kmeans.labels_
# adding labels to df
df_normalised['Labels'] = labels
# extracting cluster centres
cen = kmeans.cluster_centers_

# plotting
plt.figure(figsize=(6, 6))

# scatter plot
plt.scatter(df_normalised[df_normalised['Labels'] == 0]['1965'],
            df_normalised[df_normalised['Labels'] == 0]["2015"],
            c='red', label='Cluster1')
plt.scatter(df_normalised[df_normalised['Labels'] == 1]['1965'],
            df_normalised[df_normalised['Labels'] == 1]["2015"],
            c='blue', label='Cluster2')

# plot cluster centres
xc = cen[:, 0]
yc = cen[:, 1]
plt.scatter(xc, yc, c="k", marker="d", s=80, label='Centres')

plt.xlabel("1965")
plt.ylabel("2015")
plt.title("2 clusters")
plt.legend()
plt.show()
