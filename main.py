# This is a sample Python script.
# Import libraries
import yfinance as yf
import pandas as pd
from colorama import Fore
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def retrieving_nasdaq_information():
    with open("nasdaq_100_tickerslist.txt") as file:
        list_of_tickers = [line.rstrip('\n') for line in file]
        print(Fore.GREEN + "Pulled Ticker Symbols ✓")

        print(Fore.GREEN + "Uploading Data to a CSV File..........")
        stock_symbol_data = yf.download(tickers=list_of_tickers, period='1y', interval='1d')['Close']
        df = stock_symbol_data.T

        # Saving to a csv, overwrites the file each time
        df.to_csv("nasdaq_data.csv", mode='w')
        print(Fore.GREEN + "Uploaded Data To nasdaq_data CSV ✓")


def kmeans_clustering():
    # Loading the dataset
    dataset = pd.read_csv("nasdaq_data.csv", index_col=0)

    x = dataset.iloc[:, 0:260].values
    y = dataset.iloc[:, 100].values

    x = StandardScaler().fit_transform(x)

    # Reduce Data
    pca = PCA(n_components=12)
    PCs = pca.fit_transform(x)

    explained_variance = pca.explained_variance_ratio_

    PC_df = pd.DataFrame(PCs, index=dataset.index[:])
    PC_df.to_csv("PCAReduction.csv", mode='w')
    print(PC_df)

    # Kmeans Cluster
    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
    kmeans.fit(PC_df)
    kmeans.predict(PC_df)
    labels = kmeans.labels_

    # Getting ticker symbols in their cluster
    cluster0 = []
    cluster1 = []
    cluster2 = []
    cluster3 = []

    for x, ticker_symbol in enumerate(PC_df.index):
        kmeans_label = labels[x]
        if kmeans_label == 0:
            cluster0.append(ticker_symbol)
        elif kmeans_label == 1:
            cluster1.append(ticker_symbol)
        elif kmeans_label == 2:
            cluster2.append(ticker_symbol)
        else:
            cluster3.append(ticker_symbol)

    print('Clusters Below:')
    print(Fore.LIGHTWHITE_EX + f"Cluster 1:")
    for tickers in cluster0:
        print(f'{tickers}')

    print(Fore.BLUE + f"\nCluster 2:")
    for tickers in cluster1:
        print(f'{tickers}')

    print(Fore.YELLOW + f"\nCluster 3:")
    for tickers in cluster2:
        print(f'{tickers}')

    print(Fore.MAGENTA + f"\nCluster 4:")
    for tickers in cluster3:
        print(f'{tickers}')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    retrieving_nasdaq_information()
    kmeans_clustering()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
