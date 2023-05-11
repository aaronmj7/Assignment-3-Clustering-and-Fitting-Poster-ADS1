# -*- coding: utf-8 -*-
"""
Created on Tue May  9 22:14:14 2023

@author: aaron
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import cluster_tools as ct
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as opt
import errors as err


def read_df(fname):
    ''' Function to read a csv file in world bank format and clean it for
    clustering.
    Arguments:
        Name/file path of a csv file.
    Returns the dataframe and transposed dataframe.
    '''

    # reading from csv
    df = pd.read_csv(fname, skiprows=4)

    # dropping columns that are not needed
    df.drop(["Country Code", "Indicator Code", "Indicator Name"], axis=1,
            inplace=True)

    # setting index
    df.set_index(['Country Name'], inplace=True)
    # naming column
    df.columns.name = "Year"

    # droping columns and rows with nan values
    df.dropna(how="all", axis=1, inplace=True)
    df.dropna(how="any", thresh=50, inplace=True)

    # filling nan values
    df.fillna(0, inplace=True)

    # transposing
    df_t = df.transpose()
    # change dtype to int
    df_t.index = pd.to_numeric(df_t.index)

    # droping columns with nan values
    df_t.dropna(how='all', axis=1, inplace=True)

    return df, df_t


def elbow_method(df, n):
    ''' Function to plot Elbow method graph.
    Argument:
        A dataframe
        Maximum number of clusters to check.
    '''

    # calculate wcss for different no. of clusters
    wcss = []
    for i in range(1, n+1):
        # set up kmeans and fit
        kmeans = KMeans(n_clusters=i, max_iter=300, n_init=10)
        kmeans.fit(df)
        # append to wcss list
        wcss.append(kmeans.inertia_)

    # plotting to use elbow method
    plt.figure(figsize=(14, 7))

    plt.plot(range(1, n+1), wcss, marker='o')

    # tile and labels
    plt.title('The Elbow method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')

    # saving
    plt.savefig('Elbow method.png', dpi=650)
    # show the plot
    plt.show()

    return


def cluster_plot(df, nc):
    ''' Function to plot clustering graph.
    Arguments:
        A dataframe.
        The number of clusters needed.
    Returns the cluster centres and array of labels.

    '''

    # set up the clusterer with the number of expected clusters
    kmeans = KMeans(n_clusters=nc)
    # fit data
    kmeans.fit(df)

    # extracting labels
    labels = kmeans.labels_
    # adding labels to df
    df['Labels'] = labels

    # extracting cluster centres
    cen = kmeans.cluster_centers_

    # plotting
    plt.figure(figsize=(7, 7))

    # setting color and label
    colors = ['red', 'blue', 'green', 'orange', 'pink']
    labeling = ['Cluster '+str(i+1) for i in range(nc)]

    # plot with for loop
    for i in range(nc):
        plt.scatter(df[df['Labels'] == i]['1965'],
                    df[df['Labels'] == i]['2015'],
                    c=colors[i], label=labeling[i])

    # plot cluster centres
    xc = cen[:, 0]
    yc = cen[:, 1]

    plt.scatter(xc, yc, c="k", marker="d", s=80, label='Centres')

    # tile and labels
    plt.xlabel("1965")
    plt.ylabel("2015")
    plt.title(str(nc) + " Clusters")

    # show legend
    plt.legend()

    # saving
    plt.savefig((str(nc) + " Clusters" + '.png'), dpi=650)
    # show the plot
    plt.show()

    return cen, labels


def logistic(t, g, t0, n0):
    '''Calculates the logistic function with scale factor n0 and growth rate g
    '''

    f = n0 / (1 + np.exp(-g*(t-t0)))

    return f


# read the data
df_ogc, df_ogc_t = read_df("API_EG.ELC.FOSL.ZS_DS2_en_csv_v2_5211704.csv")

# setting seaborn theme
sns.set_theme(font_scale=1.25)


# clustering

# years for scatter matrix
years = ['1960', '1961', '1962', '1963', '1964', '1965',
         '2010', '2011', '2012', '2013', '2014', '2015']
# plotting scatter matrix to find good cluster
pd.plotting.scatter_matrix(df_ogc[years], figsize=(12, 12))
# show the plot
plt.show()

# chosing best years for clustering
df_clstr = df_ogc[['1965', '2015']].copy()

# normalising and storing minimum and maximum
df_normalised, df_min, df_max = ct.scaler(df_clstr)

# using elbow method for finding best number of cluster
elbow_method(df_normalised, 10)

# plot for three clusters
cen3, labels3 = cluster_plot(df_normalised, 3)

# plot for four clusters
cen4, labels4 = cluster_plot(df_normalised, 4)

# backscale the normalised df
cen_bckscl = ct.backscale(cen4, df_min, df_max)

# backscaled cluster graph
plt.figure(figsize=(7, 7))

plt.scatter(df_ogc["1990"], df_ogc["2015"], c=labels4, cmap='Set1')

# plot centers
xcen = cen_bckscl[:, 0]
ycen = cen_bckscl[:, 1]
plt.scatter(xcen, ycen, 45, "k", marker="d")

# label and title
plt.xlabel("1965")
plt.ylabel("2015")
plt.title("Backscaled Cluster Graph")
# show the plot
plt.show()


# fitting

# data to fit
df_tofit = pd.DataFrame(df_ogc_t.iloc[:, 3].copy())

# plot data to be fitted
df_tofit.plot(title='Electricity Produced from Oil, Gas and Coal Sources',
              ylabel='Electricity Produced(% of Total)')
# set limit
plt.xlim(1960, 2015)
# show the plot
plt.show()

# fitting logistic function with some initial values
param, covar = opt.curve_fit(logistic, df_tofit.index, df_tofit.iloc[:, 0],
                             p0=(0.0, 1987.5, 100))

# add fit values to dataframe
df_tofit["fit"] = logistic(df_tofit.index, *param)

# plot data fitting
plt.figure(figsize=(10, 5))

plt.plot(df_tofit.index, df_tofit.iloc[:, 0], c='dodgerblue', label="data")
plt.plot(df_tofit.index, df_tofit["fit"], c='k', label="fit")

# calculate sigma
sigmas = np.sqrt(np.diag(covar))

# extend years for prediction
years = np.arange(1960, 2041)

# calculate upper and lower limits
lower, upper = err.err_ranges(years, logistic, param, sigmas)
# plot error ranges
plt.fill_between(years, lower, upper, color='orange', alpha=0.5)
# set ticks and limit
plt.xticks(range(1960, 2041, 10))
plt.xlim(1960, 2040)
# label and title
plt.xlabel('Years')
plt.ylabel('Electricity Produced(% of Total)')
title = "Fitting Canada's Electricity Produced from Oil, Gas and Coal Sources"
plt.title(title)
# show legend
plt.legend()
# show the plot
plt.show()
