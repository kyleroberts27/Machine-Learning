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
from operator import itemgetter
import plotly.express as px

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

chosen_symbol_list = ['SBUX', 'MELI', 'BKNG', 'CTAS']

correlated_stocks = pd.read_csv("correlation.csv")
original_nasdaq_data = pd.read_csv("nasdaq_data_original.csv")


def retrieving_nasdaq_information():
    with open("nasdaq_100_tickerslist.txt") as file:
        list_of_tickers = [line.rstrip('\n') for line in file]
        print(Fore.GREEN + "Pulled Ticker Symbols ✓")

        print(Fore.GREEN + "Uploading Data to a CSV File.........." + Fore.LIGHTWHITE_EX)
        stock_symbol_data = yf.download(tickers=list_of_tickers, period='1y', interval='1d')['Close']
        df = stock_symbol_data.T
        df2 = stock_symbol_data

        # Saving to a csv, overwrites the file each time
        df.to_csv("nasdaq_data.csv", mode='w')
        df2.to_csv("nasdaq_data_original.csv", mode='w')
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
    print(Fore.LIGHTWHITE_EX + f'{PC_df}')

    print(Fore.GREEN + "PCA Reduction Complete ✓")

    # Kmeans Cluster
    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
    kmeans.fit(PC_df)
    kmeans.predict(PC_df)
    labels = kmeans.labels_

    print("Kmeans Clustering Complete ✓")

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

    print(Fore.GREEN + f"\nTicker Clusters Shown ✓")


def stock_correlation():
    total_cols = 100
    df = pd.read_csv("nasdaq_data_original.csv", usecols=range(1, total_cols))
    correlation = df.corr()
    correlation.to_csv("correlation.csv", mode='w')
    print(Fore.GREEN + f"Stock Correlation Complete ✓\n")


def correlation_chosen_stocks(chosen_ticker, row1, row2):
    # Putting the data into a dictionary
    ticker_dictionary = {}
    for key in row1:
        for value in row2:
            ticker_dictionary[key] = value
            row2.remove(value)
            break

    # Initialize the largest_stocks and sets it to 11
    largest_stocks = 11

    # Get the 11 largest values in dictionary
    # Using sorted() + itemgetter() + items()
    result_values_largest = dict(sorted(ticker_dictionary.items(), key=itemgetter(1), reverse=True)[:largest_stocks])

    # Removes the own stock from the dictionary as it is the most correlated
    del result_values_largest[chosen_ticker]

    # Printing the result
    print(Fore.LIGHTWHITE_EX + f"The top 10 positively correlated stocks for {chosen_ticker}:")

    counter = 1

    # Printing a dictionary using a loop and the items() method
    for key, value in result_values_largest.items():
        print(f"{counter}. {key}: {value}")
        counter += 1
    print("\n")

    # Initialize the smallest_stocks and sets it to 11
    smallest_stocks = 10

    # The 10 smallest values in dictionary
    # Using sorted() + itemgetter() + items()
    result_values_smallest = dict(sorted(ticker_dictionary.items(), key=itemgetter(1), reverse=False)[:smallest_stocks])

    # Printing the result
    print(f"The top 10 negatively correlated stocks for {chosen_ticker}:")

    counter = 1

    # Printing a dictionary using a loop and the items() method
    for key, value in result_values_smallest.items():
        print(f"{counter}. {key}: {value}")
        counter += 1
    print("\n")


def sbux_correlated_stocks():
    # Getting data
    row1 = correlated_stocks.iloc[0:, 0].tolist()
    row2 = correlated_stocks['SBUX'].tolist()
    chosen_ticker = chosen_symbol_list[0]

    correlation_chosen_stocks(chosen_ticker, row1, row2)


def meli_correlated_stocks():
    # Getting data
    df = pd.read_csv("correlation.csv")
    row1 = df.iloc[0:, 0].tolist()
    row2 = df['MELI'].tolist()
    chosen_ticker = chosen_symbol_list[1]

    correlation_chosen_stocks(chosen_ticker, row1, row2)


def bkng_correlated_stocks():
    # Getting data
    df = pd.read_csv("correlation.csv")
    row1 = df.iloc[0:, 0].tolist()
    row2 = df['BKNG'].tolist()
    chosen_ticker = chosen_symbol_list[2]

    correlation_chosen_stocks(chosen_ticker, row1, row2)


def ctas_correlated_stocks():
    # Getting data
    df = pd.read_csv("correlation.csv")
    row1 = df.iloc[0:, 0].tolist()
    row2 = df['CTAS'].tolist()
    chosen_ticker = chosen_symbol_list[3]

    correlation_chosen_stocks(chosen_ticker, row1, row2)


def eda_analysis_stocks(chosen_ticker, x_data, y_data):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=x_data, y=y_data, data=original_nasdaq_data, color='blue')
    plt.title(f'Closing Stock Price of {chosen_ticker} Over Time')
    plt.xlabel('Date')
    plt.xticks(x_data, rotation='vertical')
    plt.ylabel('Closing Stock Price')

    # Setting the number of ticks
    plt.locator_params(axis='x', nbins=24)


def sbux_eda_analysis_stocks():
    chosen_ticker = chosen_symbol_list[0]
    x_data = original_nasdaq_data.iloc[:, 0]
    y_data = original_nasdaq_data[chosen_ticker]
    eda_analysis_stocks(chosen_ticker, x_data, y_data)


def meli_eda_analysis_stocks():
    chosen_ticker = chosen_symbol_list[1]
    x_data = original_nasdaq_data.iloc[:, 0]
    y_data = original_nasdaq_data[chosen_ticker]
    eda_analysis_stocks(chosen_ticker, x_data, y_data)


def bkng_eda_analysis_stocks():
    chosen_ticker = chosen_symbol_list[2]
    x_data = original_nasdaq_data.iloc[:, 0]
    y_data = original_nasdaq_data[chosen_ticker]
    eda_analysis_stocks(chosen_ticker, x_data, y_data)


def ctas_eda_analysis_stocks():
    chosen_ticker = chosen_symbol_list[3]
    x_data = original_nasdaq_data.iloc[:, 0]
    y_data = original_nasdaq_data[chosen_ticker]
    eda_analysis_stocks(chosen_ticker, x_data, y_data)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #retrieving_nasdaq_information()
    #kmeans_clustering()
    #stock_correlation()
    sbux_correlated_stocks()
    meli_correlated_stocks()
    bkng_correlated_stocks()
    ctas_correlated_stocks()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
