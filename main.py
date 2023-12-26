# This is a sample Python script.
# Import libraries
import yfinance as yf
import pandas as pd
from pandas import to_datetime
from colorama import Fore
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
from operator import itemgetter
import plotly.express as px
from prophet import Prophet
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from pmdarima.arima import auto_arima
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

chosen_symbol_list = ['SBUX', 'MELI', 'BKNG', 'CTAS']

correlated_stocks = pd.read_csv("correlation.csv")
original_nasdaq_data = pd.read_csv("nasdaq_data_original.csv")
prophet_data = pd.read_csv("nasdaq_data_original.csv")
dateparse = lambda dates: pd.to_datetime(dates, format='%Y-%m-%d')
arima_data = pd.read_csv('nasdaq_data_original.csv',sep=',', index_col='Date', parse_dates=['Date'], date_parser=dateparse).fillna(0)


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


def arima_stocks(chosen_stock, stock_symbol):
    train_data, test_data = chosen_stock[3:int(len(chosen_stock) * 0.05)], chosen_stock[int(len(chosen_stock) * 0.05):]

    train_arima = train_data
    test_arima = test_data

    history = [x for x in train_arima]
    y = test_arima
    # make first prediction
    predictions = list()
    model = sm.tsa.arima.ARIMA(history, order=(1, 1, 1))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
    history.append(y[0])

    # rolling forecasts
    for i in range(1, len(y)):
        # predict
        model = sm.tsa.arima.ARIMA(history, order=(1, 1, 0))
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        # invert transformed prediction
        predictions.append(yhat)
        # observation
        obs = y[i]
        history.append(obs)

    # report performance
    print(f'{stock_symbol}:')
    mse = mean_squared_error(y, predictions)
    print('MSE: ' + str(mse))
    mae = mean_absolute_error(y, predictions)
    print('MAE: ' + str(mae))
    rmse = math.sqrt(mean_squared_error(y, predictions))
    print('RMSE: ' + str(rmse))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(16, 8))
    plt.plot(chosen_stock.index[-600:], chosen_stock.tail(600), color='green', label='Train Stock Price')
    plt.plot(test_data.index, y, color='blue', label='Real Stock Price')
    plt.plot(test_data.index, predictions, color='red', label='Predicted Stock Price')
    plt.title(f'{stock_symbol} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel(f'{stock_symbol} Stock Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('arima_model.pdf')
    plt.show()


def sbux_arima():
    chosen_stock = arima_data['SBUX']
    stock_symbol = chosen_symbol_list[0]
    arima_stocks(chosen_stock, stock_symbol)


def meli_arima():
    chosen_stock = arima_data['MELI']
    stock_symbol = chosen_symbol_list[1]
    arima_stocks(chosen_stock, stock_symbol)


def bkng_arima():
    chosen_stock = arima_data['BKNG']
    stock_symbol = chosen_symbol_list[2]
    arima_stocks(chosen_stock, stock_symbol)


def ctas_arima():
    chosen_stock = arima_data['CTAS']
    stock_symbol = chosen_symbol_list[3]
    arima_stocks(chosen_stock, stock_symbol)


def linear_regression(dates, prices, chosen_stock):
    # Convert to numpy array and reshape them
    dates = np.asanyarray(dates)
    prices = np.asanyarray(prices)
    dates = np.reshape(dates, (len(dates), 1))
    prices = np.reshape(prices, (len(prices), 1))

    # Load Pickle File to get the previous saved model accuracy
    try:
        pickle_in = open("prediction.pickle", "rb")
        reg = pickle.load(pickle_in)
        xtrain, xtest, ytrain, ytest = train_test_split(dates, prices, test_size=1)
        best = reg.score(ytrain, ytest)
    except:
        pass

    # Get the highest accuracy model
    best = 0
    for _ in range(100):
        xtrain, xtest, ytrain, ytest = train_test_split(dates, prices, test_size=0.80)
        reg = LinearRegression().fit(xtrain, ytrain)
        acc = reg.score(xtest, ytest)
        if acc > best:
            best = acc
            # Save model to pickle format
            with open('prediction.pickle', 'wb') as f:
                pickle.dump(reg, f)
            print(acc)

    # Load linear regression model
    pickle_in = open("prediction.pickle", "rb")
    reg = pickle.load(pickle_in)

    # Get the average accuracy of the model
    mean = 0
    for i in range(10):
        # Random Split Data
        msk = np.random.rand(len(original_nasdaq_data)) < 0.8
        xtest = dates[~msk]
        ytest = prices[~msk]
        mean += reg.score(xtest, ytest)

    print("Average Accuracy:", mean / 10)

    # Plot Predicted VS Actual Data
    plt.plot(xtest, ytest, color='green', linewidth=1, label='Actual Price')  # plotting the initial datapoints
    plt.plot(xtest, reg.predict(xtest), color='blue', linewidth=3,
             label='Predicted Price')  # plotting the line made by linear regression
    plt.title(f"Linear Regression {chosen_stock} | Time vs. Price ")
    plt.legend()
    plt.xlabel('Date Integer')
    plt.show()


def sbux_linear_regression():
    dates = list(range(0, int(len(original_nasdaq_data))))
    chosen_stock = chosen_symbol_list[0]
    prices = original_nasdaq_data[chosen_stock]
    linear_regression(dates, prices, chosen_stock)


def meli_linear_regression():
    dates = list(range(0, int(len(original_nasdaq_data))))
    chosen_stock = chosen_symbol_list[1]
    prices = original_nasdaq_data[chosen_stock]
    linear_regression(dates, prices, chosen_stock)


def bkng_linear_regression():
    dates = list(range(0, int(len(original_nasdaq_data))))
    chosen_stock = chosen_symbol_list[2]
    prices = original_nasdaq_data[chosen_stock]
    linear_regression(dates, prices, chosen_stock)


def ctas_linear_regression():
    dates = list(range(0, int(len(original_nasdaq_data))))
    chosen_stock = chosen_symbol_list[3]
    prices = original_nasdaq_data[chosen_stock]
    linear_regression(dates, prices, chosen_stock)


def prophet_analysis(chosen_stock, days):
    df = prophet_data.reset_index().rename(columns={'Date': 'ds', chosen_stock: 'y'})
    df['y'] = np.log(df['y'])
    model = Prophet(
        daily_seasonality=True,
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0
    )

    model.fit(df)
    future = model.make_future_dataframe(periods=days)  # forecasting for 1 year from now.
    forecast = model.predict(future)

    figure = model.plot(forecast)
    plt.title(f"Facebook Prediction for {chosen_stock}")
    plt.show()


def sbux_prophet():
    chosen_stock = chosen_symbol_list[0]
    days = 365
    prophet_analysis(chosen_stock, days)


def meli_prophet():
    chosen_stock = chosen_symbol_list[1]
    days = 365
    prophet_analysis(chosen_stock, days)


def bkng_prophet():
    chosen_stock = chosen_symbol_list[2]
    days = 365
    prophet_analysis(chosen_stock, days)


def ctas_prophet():
    chosen_stock = chosen_symbol_list[3]
    days = 365
    prophet_analysis(chosen_stock, days)


def stock_correlation():
    total_cols = 100
    df = pd.read_csv("nasdaq_data_original.csv", usecols=range(1, total_cols))
    correlation = df.corr()
    correlation.to_csv("correlation.csv", mode='w')
    print(Fore.GREEN + f"Stock Correlation Complete ✓\n")


def positive_correlation_stocks(chosen_ticker, row1, row2):
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


def negative_stock_correlation(chosen_ticker, row1, row2):
    # Putting the data into a dictionary
    ticker_dictionary = {}
    for key in row1:
        for value in row2:
            ticker_dictionary[key] = value
            row2.remove(value)
            break

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


def sbux_neg_correlation():
    # Getting data
    row1 = correlated_stocks.iloc[0:, 0].tolist()
    row2 = correlated_stocks['SBUX'].tolist()
    chosen_ticker = chosen_symbol_list[0]

    negative_stock_correlation(chosen_ticker, row1, row2)


def meli_neg_correlation():
    # Getting data
    row1 = correlated_stocks.iloc[0:, 0].tolist()
    row2 = correlated_stocks['MELI'].tolist()
    chosen_ticker = chosen_symbol_list[1]

    negative_stock_correlation(chosen_ticker, row1, row2)



def bkng_neg_correlation():
    # Getting data
    row1 = correlated_stocks.iloc[0:, 0].tolist()
    row2 = correlated_stocks['BKNG'].tolist()
    chosen_ticker = chosen_symbol_list[2]

    negative_stock_correlation(chosen_ticker, row1, row2)


def ctas_neg_correlation():
    # Getting data
    row1 = correlated_stocks.iloc[0:, 0].tolist()
    row2 = correlated_stocks['CTAS'].tolist()
    chosen_ticker = chosen_symbol_list[3]

    negative_stock_correlation(chosen_ticker, row1, row2)



def pos_sbux_correlated_stocks():
    # Getting data
    row1 = correlated_stocks.iloc[0:, 0].tolist()
    row2 = correlated_stocks['SBUX'].tolist()
    chosen_ticker = chosen_symbol_list[0]

    positive_correlation_stocks(chosen_ticker, row1, row2)


def pos_meli_correlated_stocks():
    # Getting data
    df = pd.read_csv("correlation.csv")
    row1 = df.iloc[0:, 0].tolist()
    row2 = df['MELI'].tolist()
    chosen_ticker = chosen_symbol_list[1]

    positive_correlation_stocks(chosen_ticker, row1, row2)


def pos_bkng_correlated_stocks():
    # Getting data
    df = pd.read_csv("correlation.csv")
    row1 = df.iloc[0:, 0].tolist()
    row2 = df['BKNG'].tolist()
    chosen_ticker = chosen_symbol_list[2]

    positive_correlation_stocks(chosen_ticker, row1, row2)


def pos_ctas_correlated_stocks():
    # Getting data
    df = pd.read_csv("correlation.csv")
    row1 = df.iloc[0:, 0].tolist()
    row2 = df['CTAS'].tolist()
    chosen_ticker = chosen_symbol_list[3]

    positive_correlation_stocks(chosen_ticker, row1, row2)


def eda_analysis_stocks(chosen_ticker, x_data, y_data):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=x_data, y=y_data, data=original_nasdaq_data, color='blue')
    plt.title(f'Closing Stock Price of {chosen_ticker} Over Time')
    plt.xlabel('Date')
    plt.xticks(x_data, rotation='vertical')
    plt.ylabel('Closing Stock Price')

    # Setting the number of ticks
    plt.locator_params(axis='x', nbins=24)
    plt.show()


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
    #pos_sbux_correlated_stocks()
    # pos_meli_correlated_stocks()
    # pos_bkng_correlated_stocks()
    # pos_ctas_correlated_stocks()
    # sbux_eda_analysis_stocks()
    # meli_eda_analysis_stocks()
    # bkng_eda_analysis_stocks()
    # ctas_eda_analysis_stocks()
    # sbux_prophet()
    # meli_prophet()
    # bkng_prophet()
    #ctas_prophet()
    #sbux_linear_regression()
    #meli_linear_regression()
    #bkng_linear_regression()
    #ctas_linear_regression()
    # sbux_neg_correlation()
    # meli_neg_correlation()
    # bkng_neg_correlation()
    # ctas_neg_correlation()
    sbux_arima()
    meli_arima()
    bkng_arima()
    ctas_arima()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/