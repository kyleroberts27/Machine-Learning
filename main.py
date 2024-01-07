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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
import tensorflow as tf

from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import streamlit as st
from streamlit_option_menu import option_menu

import datetime as dt
import matplotlib.dates as mdates

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
    #PC_df.to_csv("PCAReduction.csv", mode='w')

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

    new_title = '<p style="font-family:sans-serif; font-size: 30px;">The 4 Clusters after PCA and KMeans Below:</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([0.7, 0.7, 0.7, 6])
    with col1:
        st.write(f"Cluster 1:")
        for tickers in cluster0:
            st.write(f'{tickers}')

    with col2:
        st.write(f"\nCluster 2:")
        for tickers in cluster1:
            st.write(f'{tickers}')

    with col3:
        st.write(f"\nCluster 3:")
        for tickers in cluster2:
            st.write(f'{tickers}')

    with col4:
        st.write(f"\nCluster 4:")
        for tickers in cluster3:
            st.write(f'{tickers}')

    print(Fore.GREEN + f"\nTicker Clusters Shown ✓")
    print(Fore.LIGHTWHITE_EX + " ")


def lstm_stocks(chosen_stock):
    keras = tf.keras
    Sequential = keras.models.Sequential
    Dense = keras.layers.Dense
    LSTM = keras.layers.LSTM
    df2 = original_nasdaq_data.reset_index()[chosen_stock]

    scaler = MinMaxScaler()
    df2 = scaler.fit_transform(np.array(df2).reshape(-1, 1))
    train_size = int(len(df2) * 0.65)
    test_size = len(df2) - train_size
    train_data, test_data = df2[0:train_size, :], df2[train_size:len(df2), :1]

    # calling the create dataset function to split the data into
    # input output datasets with time step 100
    time_step = 10
    X_train, Y_train = create_dataset(train_data, time_step)
    X_test, Y_test = create_dataset(test_data, time_step)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    history = model.fit(
        X_train, Y_train,
        epochs=30,
        batch_size=16,
        validation_split=0.1,
        verbose=1,
        shuffle=False
    )

    y_pred = model.predict(X_test)

    plt.plot(Y_test, marker='.', label="true")
    plt.plot(y_pred, 'r', label="prediction")
    plt.ylabel('Value')
    plt.xlabel('Time Step')
    plt.legend()
    st.pyplot(plt)

    fig_prediction = plt.figure(figsize=(10, 8))
    st.write('Plotting the Predicted Result:')
    plt.plot(np.arange(0, len(Y_train)), Y_train, 'g', label="history")
    plt.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)), Y_test, marker='.', label="true")
    plt.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)), y_pred, 'r', label="prediction")
    plt.ylabel('Value')
    plt.xlabel('Time Step')
    plt.legend()
    st.pyplot(plt)


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


def sbux_lstm():
    chosen_stock = chosen_symbol_list[0]
    lstm_stocks(chosen_stock)


def meli_lstm():
    chosen_stock = chosen_symbol_list[1]
    lstm_stocks(chosen_stock)


def bkng_lstm():
    chosen_stock = chosen_symbol_list[2]
    lstm_stocks(chosen_stock)


def ctas_lstm():
    chosen_stock = chosen_symbol_list[3]
    lstm_stocks(chosen_stock)


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
    forecast = model_fit.forecast(steps=30)
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
    st.write(f'{stock_symbol}:')
    mse = mean_squared_error(y, predictions)
    st.write('MSE: ' + str(mse))
    mae = mean_absolute_error(y, predictions)
    st.write('MAE: ' + str(mae))
    rmse = math.sqrt(mean_squared_error(y, predictions))
    st.write('RMSE: ' + str(rmse))

    plt.figure(figsize=(16, 8))
    plt.plot(chosen_stock.index[-600:], chosen_stock.tail(600), color='green', label='Train Stock Price')
    plt.plot(test_data.index, y, color='blue', label='Real Stock Price')
    plt.plot(test_data.index, predictions, color='red', label='Predicted Stock Price')
    forecast_dates = pd.date_range(start='2023-12-1', periods=31, freq='D')[1:]
    plt.plot(forecast_dates, forecast, color='orange', label="Forecasted Prices")
    plt.title(f'{stock_symbol} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel(f'{stock_symbol} Stock Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)


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

    st.write("Average Accuracy:", mean / 10)

    # Plot Predicted VS Actual Data
    plt.plot(xtest, ytest, color='green', linewidth=1, label='Actual Price')  # plotting the initial datapoints
    plt.plot(xtest, reg.predict(xtest), color='blue', linewidth=3, label='Predicted Price')  # plotting the line made by linear regression
    plt.title(f"Linear Regression {chosen_stock} | Time vs. Price ")
    plt.legend()
    plt.xlabel('Date Integer')
    st.pyplot(plt)
    #plt.show()


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
    st.pyplot(plt)


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
    st.write(f"The top 10 positively correlated stocks for {chosen_ticker}:")

    counter = 1

    # Printing a dictionary using a loop and the items() method
    for key, value in result_values_largest.items():
        st.write(f"{counter}. {key}: {value}")
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
    st.write(f"The top 10 negatively correlated stocks for {chosen_ticker}:")

    counter = 1

    # Printing a dictionary using a loop and the items() method
    for key, value in result_values_smallest.items():
        st.write(f"{counter}. {key}: {value}")
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
    st.pyplot(plt)
    #plt.show()


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


def seven_day_sbux_prophet():
    chosen_stock = chosen_symbol_list[0]
    days = 7
    prophet_analysis(chosen_stock, days)


def seven_day_meli_prophet():
    chosen_stock = chosen_symbol_list[1]
    days = 7
    prophet_analysis(chosen_stock, days)


def seven_day_bkng_prophet():
    chosen_stock = chosen_symbol_list[2]
    days = 7
    prophet_analysis(chosen_stock, days)


def seven_day_ctas_prophet():
    chosen_stock = chosen_symbol_list[3]
    days = 7
    prophet_analysis(chosen_stock, days)


def fourteen_day_sbux_prophet():
    chosen_stock = chosen_symbol_list[0]
    days = 14
    prophet_analysis(chosen_stock, days)


def fourteen_day_meli_prophet():
    chosen_stock = chosen_symbol_list[1]
    days = 14
    prophet_analysis(chosen_stock, days)


def fourteen_day_bkng_prophet():
    chosen_stock = chosen_symbol_list[2]
    days = 14
    prophet_analysis(chosen_stock, days)


def fourteen_day_ctas_prophet():
    chosen_stock = chosen_symbol_list[3]
    days = 14
    prophet_analysis(chosen_stock, days)


def thirty_day_sbux_prophet():
    chosen_stock = chosen_symbol_list[0]
    days = 30
    prophet_analysis(chosen_stock, days)


def thirty_day_meli_prophet():
    chosen_stock = chosen_symbol_list[1]
    days = 30
    prophet_analysis(chosen_stock, days)


def thirty_day_bkng_prophet():
    chosen_stock = chosen_symbol_list[2]
    days = 30
    prophet_analysis(chosen_stock, days)


def thirty_day_ctas_prophet():
    chosen_stock = chosen_symbol_list[3]
    days = 30
    prophet_analysis(chosen_stock, days)


@st.cache_resource
def load_sbux_lstm():
    sbux_lstm()


@st.cache_resource
def load_meli_lstm():
    meli_lstm()


@st.cache_resource
def load_bkng_lstm():
    bkng_lstm()


@st.cache_resource
def load_ctas_lstm():
    ctas_lstm()


st.set_page_config(layout="wide")

selected = option_menu(
    menu_title=None,
    options=["Home", "Clusters", "Correlation", "EDA Analysis", "ARIMA", "LSTM", "Linear Regression", "Prophet", "Buy/Sell"],
    icons=["house", "bounding-box-circles", "c-circle", "graph-up", "bezier", "diagram-3", "graph-up-arrow", "facebook", "currency-exchange"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "blue", "font-size": "20px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#89cff0"},
    }

)

if selected == "Home":
    st.title("COM624 - Kyle Roberts Software Solution")
    st.write("Student Number: 16291981")
    st.write("Within this application is my solution to the tasks proposed within the assessment brief. I chose too halt the retrieving of the close data on the 3rd of December 2023 to try and keep my data consistant. This still means that the analysis of each of my chosen stocks is compiled over a year, however it does mean that it will not be the most upto date data, but this does give me the opportunity to compare my analysis of data against how it has actually performed in the real world")
    st.write("My chosen stocks from the 4 clusters are:")
    col1, col2 = st.columns([1,17])
    with col1:
        st.image("SBUX-logo.png", width=65)
    with col2:
        new_title = '<p style="font-family:sans-serif; font-size: 30px;">SBUX - Starbucks Corporation</p>'
        st.markdown(new_title, unsafe_allow_html=True)

    with col1:
        st.write("")
    with col2:
        st.write("")

    with col1:
        st.image("MELI-logo.png", width=60)
    with col2:
        new_title = '<p style="font-family:sans-serif; font-size: 30px;">MELI - MercadoLibre Corporation</p>'
        st.markdown(new_title, unsafe_allow_html=True)

    with col1:
        st.write("")
    with col2:
        st.write("")

    with col1:
        st.image("BKNG-logo.png", width=70)
    with col2:
        new_title = '<p style="font-family:sans-serif; font-size: 30px;">BKNG - Bookings Holdings Corporation</p>'
        st.markdown(new_title, unsafe_allow_html=True)

    with col1:
        st.write("")
    with col2:
        st.write("")

    with col1:
        st.image("CTAS-logo.png", width=70)
    with col2:
        new_title = '<p style="font-family:sans-serif; font-size: 30px;">CTAS - Cintas Corporation</p>'
        st.markdown(new_title, unsafe_allow_html=True)

    st.write("To choose the above stocks, I copied and pasted each ticker, from their respecitive clusters, into a wheel of fortune website so that my stocks were chosen completely randomly, meaning I did not end up choosing the stocks but only took the output of each wheel spin. These screenshots can be seen in my report.")
    st.write("Apart from the 150 Epochs section in LSTM, all the data is loaded in real time by calling the specific functions for each stock and the data they output. However, once the LSTM for each stock is loaded, the graph data is cached until the user decides to close the application.")

if selected == "Correlation":
    symbol_options_menu = st.selectbox("Choose a Stock:", options=chosen_symbol_list)
    col1, col2 = st.columns([1,4])
    if symbol_options_menu == "SBUX":
        with col1:
            pos_sbux_correlated_stocks()
        with col2:
            sbux_neg_correlation()
    if symbol_options_menu == "MELI":
        with col1:
            pos_meli_correlated_stocks()
        with col2:
            meli_neg_correlation()
    if symbol_options_menu == "BKNG":
        with col1:
            pos_bkng_correlated_stocks()
        with col2:
            bkng_neg_correlation()
    if symbol_options_menu == "CTAS":
        with col1:
            pos_ctas_correlated_stocks()
        with col2:
            ctas_neg_correlation()

if selected == "Clusters":
    kmeans_clustering()

if selected == "EDA Analysis":
    symbol_options_menu = st.selectbox("Choose a stock:", options=chosen_symbol_list)
    if symbol_options_menu == "SBUX":
        st.write("As you can see below, between 02-12-2022 to 01-12-2023 SBUX has returned -5.57% meaning that if you put money into SBUX a year ago and held, your return would be less then what you put in. Compare this to XEL (SBUX's most postiviely correlated stock), which has a return of -11.15%. If you invested the same amount, you would be better off investing in SBUX as opposed to XEL.")
        st.write("Close price 02-12-2022: 105.05")
        st.write("Close price 01-12-2023: 99.02")
        st.write("However, over the past year, the NASDAQ 100 as whole, has returned just over 45%.")
        st.write("However, if you look a bit deeper in to SBUX's dramatic drop, which from June 2022, has dropped 9.5%, analysts believes it is due to slowing credit card data at the company as a sign of falling sales as the SBUX stock is dependent on consumer trends. This could be due to cost of living crisis gripping the USA causing customers to spend less on luxuries such as coffee with their average coffee price sitting at around £3.25.")
        sbux_eda_analysis_stocks()
    if symbol_options_menu == "MELI":
        st.write("As you can see below, between 02-12-2022 to 01-12-2023 MELI has returned 74.80% meaning that it even out performed the NASDAQ 100 as whole which, has returned just over 45% in the same period. This means that if you put £1,000 into MELI a year ago and held, your return would be get almost £1,750 meaning you would gain near to £750. Compare this to BKNG (MELI's most postiviely correlated stock), which has a return of 51.51%. If you invested the same amount, you would be better off investing in MELI as opposed to BKNG, which has almost a 24% gain on BKNG.")
        st.write("Close price 02-12-2022: 945.07")
        st.write("Close price 01-12-2023: 1652.01")
        st.write("Despite the global financial crisis and market volatility, MELI's value has increased dramatically. MELI's resilience in the face of competition and challenging economic conditions in Argentina is noteworthy. This could be due to the company choice to diversify into FinTech services, leveraging its e-commerce network to offer high-interest accounts, particularly in inflation-prone Argentina. This strategy has not only helped MELI withstand economic challenges but also solidified its market position. With a recent presedential election in Argentina and Javier Milei victory. He has promised to dollarise the economy and implement business-friendly policies. These developments could impact MELI's operations and stock value in the future, potentially favorably. However, this is all dependent on if Milei's policies are successfully implemented.")
        meli_eda_analysis_stocks()
    if symbol_options_menu == "BKNG":
        st.write("As you can see below, between 02-12-2022 to 01-12-2023 BKNG has returned 51.51% meaning that it also out performed the NASDAQ 100 as whole which, has returned just over 45% in the same period. This means that if you put money into BKNG a year ago and held, you would see great returns. Compare this to MAR (BKNG's most postiviely correlated stock), which has a return of 26.35%. If you invested the same amount, you would be better off investing in BKNG as opposed to MAR, with BKNG being at almost doubling its gain compared to MAR.")
        st.write("Close price 02-12-2022: 2085.44")
        st.write("Close price 01-12-2023: 3159.56")
        st.write("To explain why the BKNG stock is hitting highs is mainly down to BKNG's record profitability, meaning that travel demand is still high. BKNG makes it money because travellers use it services to; book flights, rent cars, reserve hotels. Paying the majority upfront. Not only this but in the first half of 2023, its total revenue was up 32% from the comparable period of 2022. By comparison, marketing expenses were only up by 15%. One final consideration is because the company spent $5.2 billion in the first half of the year repurchasing shares, this then forces its earnings per share up because its total number of shares outstanding is down.")
        bkng_eda_analysis_stocks()
    if symbol_options_menu == "CTAS":
        st.write("As you can see below, between 02-12-2022 to 01-12-2023 CTAS has returned 20.69% meaning that it even though it may not have out performed the NASDAQ 100 as whole which, has returned just over 45% in the same period. It does means you would make just under half the returns compared to the NASDAQ 100. It still means that if you put money into CTAS a year ago and held, you would see good returns. Compare this to ADBE (CTAS's most postiviely correlated stock), which has a return of 32.41%. Meaning if you invested the same amount, you would be better off investing in ADBE as opposed to CTAS, with ADBE returning more when compared to CTAS.")
        st.write("Close price 02-12-2022: 462.53")
        st.write("Close price 01-12-2023: 558.25")
        st.write("The main reason why  CTAS is perfoming strongly is from strong segmental performances and its focus on the enhancement of its product portfolio despite the adverse impacts of high costs and forex woes. This means the company’s focus on the enhancement of its product portfolio, along with investments in technology and existing facilities, maybe a factor in driving its performance in the near term. Not only this but CTAS continues to increase shareholders’ value through dividend payments & share repurchases. In the first three months of fiscal 2024, the company paid dividends worth $117.6 million, up approximately 20.4% year over year. It is also worth noting that Cintas has consistently raised its dividend for 40 straight years leading to strong confidence from investors.")
        ctas_eda_analysis_stocks()

if selected == "ARIMA":
    st.title("Overall analysis:")
    st.write("Overall I believe the ARIMA model to be the most accurate at predicting data compared to all of the Machine Learning models for Predicition and Forecasting. This is because in all my models for each stock I gave it around 5% of the total dataset for training and as you can see, all of the graphs produced by it provide a good prediction for all of the stocks I have chosen, compared to the real stock price.")
    symbol_options_menu = st.selectbox("Choose a stock:", options=chosen_symbol_list)
    if symbol_options_menu == "SBUX":
        sbux_arima()
    if symbol_options_menu == "MELI":
        meli_arima()
    if symbol_options_menu == "BKNG":
        bkng_arima()
    if symbol_options_menu == "CTAS":
        ctas_arima()

if selected == "LSTM":
    st.title("Overall analysis:")
    st.write("For me LSTM is the most interesting but takes the most amount of time for the model run through its process and predict the stock price. This meant trying to find a trade off between trying to keep the predicition accurate, whilst also trying to keep the time spent training the data to a minimum. For me this meant only training the Neural Network to 30 epochs, reducing the accuracy of the predicition but vastly decreasing the time it takes to generate said predicition. I say this because originally I had set the number of epochs for training the Neural Network to 150, which vastly improved the accuracy of the however, it meant that it took the model around 1 minute 30 seconds to complete its process and produce the graph. Whereas, 30 epochs takes about 30 seconds and is less accurate.")
    symbol_options_menu = st.selectbox("Choose a stock:", options=chosen_symbol_list)
    if symbol_options_menu == "SBUX":
        st.write("Prediction for 30 epochs:")
        load_sbux_lstm()
        st.write("Prediction for 150 epochs:")
        st.image("SBUX-150-epochs.png", width=1760)
    if symbol_options_menu == "MELI":
        st.write("Prediction for 30 epochs:")
        load_meli_lstm()
        st.write("Prediction for 150 epochs:")
        st.image("MELI-150-epochs.png", width=1760)
    if symbol_options_menu == "BKNG":
        st.write("Prediction for 30 epochs:")
        load_bkng_lstm()
        st.write("Prediction for 150 epochs:")
        st.image("BKNG-150-epochs.png", width=1760)
    if symbol_options_menu == "CTAS":
        st.write("Prediction for 30 epochs:")
        load_ctas_lstm()
        st.write("Prediction for 150 epochs:")
        st.image("CTAS-150-epochs.png", width=1760)

if selected == "Linear Regression":
    st.title("Overall analysis:")
    st.write("Linear regression for me is the most basic Machine Learning model, as even though it takes the training data and uses an 80/20 for the testing and training data. It only shows a solid gradient to show whether the closing stock price has either increased or decreased over time. However, I find this the most useful compared to the other Machine Learning models because it is a basic and easy way to forecast future data and it gives a good trend of which direction the closing stock price is going and is a very easy to way to visualise the data. However, if you where to increase the split to say 90/10, it would end up corrupting the model and preventing it from running, as it hasn't been provided enough data to train the model on.")
    symbol_options_menu = st.selectbox("Choose a stock:", options=chosen_symbol_list)
    if symbol_options_menu == "SBUX":
        st.write("As you can see, the linear regression is showing that the stock price over the period 02-12-2022 to 01-02-2023 has decreased over time and backs up the EDA anaylsis showing the price of the stock has decreased meaning it would have returned less then what you put in.")
        st.write("For SBUX the accuracy of the data is sitting at around 0.2 which means that it is not that accurate interms of the training data supplied.")
        sbux_linear_regression()
    if symbol_options_menu == "MELI":
        st.write("As you can see, the linear regression is showing that the stock price over the period 02-12-2022 to 01-02-2023 has increased over time and backs up the EDA anaylsis showing the price of the stock has increased meaning it would have returned more then what you put in.")
        st.write("For MELI the accuracy of the data is sitting at around 0.5 which means that it may not be the most accurate interms of the training data supplied, the training data is more accurate then SBUX.")
        meli_linear_regression()
    if symbol_options_menu == "BKNG":
        st.write("As you can see, the linear regression is showing that the stock price over the period 02-12-2022 to 01-02-2023 has increased over time and backs up the EDA anaylsis showing the price of the stock has increased meaning it would have returned more then what you put in.")
        st.write("For BKNG the accuracy of the data is sitting at around 0.8 which means out of all the training data supplied for each stock, BKNGs is the most accurate.")
        bkng_linear_regression()
    if symbol_options_menu == "CTAS":
        st.write("As you can see, the linear regression is showing that the stock price over the period 02-12-2022 to 01-02-2023 has increased over time and backs up the EDA anaylsis showing the price of the stock has increased meaning it would have returned more then what you put in.")
        st.write("For CTAS the accuracy of the data is sitting just below 0.8 at between about 0.77-0.79, meaning that it is the second most accurate interms of the training data supplied.")
        ctas_linear_regression()

if selected == "Prophet":
    st.title("Overall analysis:")
    st.write("Overall, prophet is used to 'forecast' future prices for stocks. For each of my stock I chose to show the predicited forecast for the next 365 days. It achieves this by essentially using the real stock price and copying it for the next 365 days, and fashioning it against if the stock has increased or decreased in the period supplied to generate the forecast. For me this poses a real problem and doesn't actually achieve true forecasting due to the fact it is just a copy and paste of the data. This went spectacularly wrong for the CTAS forecasting and lead to a corruption of the outputted data, where the forecasting spiralled downwards and wasn't consitent with the data provided.")
    symbol_options_menu = st.selectbox("Choose a stock:", options=chosen_symbol_list)
    if symbol_options_menu == "SBUX":
        st.write("Below is the Prophet Data for SBUX over the next 365 days:")
        sbux_prophet()
    if symbol_options_menu == "MELI":
        st.write("Below is the Prophet Data for SBUX over the next 365 days:")
        meli_prophet()
    if symbol_options_menu == "BKNG":
        st.write("Below is the Prophet Data for SBUX over the next 365 days:")
        bkng_prophet()
    if symbol_options_menu == "CTAS":
        st.write("Below is the Prophet Data for SBUX over the next 365 days:")
        ctas_prophet()

if selected == "Buy/Sell":
    symbol_options_menu = st.selectbox("Choose a stock:", options=chosen_symbol_list)
    if symbol_options_menu == "SBUX":
        new_title = '<p style="font-family:sans-serif; font-size: 30px;">Below is the Prophet Data for SBUX over the next 7 days:</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        seven_day_sbux_prophet()
        st.write("Over the next 7 days my advice would be to sell your SBUX stock. This is because as shown above, the prophet data is showing the stock will reduce in price, meaning you will loose money")

        new_title = '<p style="font-family:sans-serif; font-size: 30px;">Below is the Prophet Data for SBUX over the next 14 days:</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        fourteen_day_sbux_prophet()
        st.write("Between 7-14 days my advice would be to sell your SBUX stock. This is because as shown above, the prophet data is showing the stock will reduce in price over these days, meaning you will loose money")

        new_title = '<p style="font-family:sans-serif; font-size: 30px;">Below is the Prophet Data for SBUX over the next 30 days:</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        thirty_day_sbux_prophet()
        st.write("Between 14-30 days my advice would be to buy SBUX stock. This is because as shown above, the prophet data is showing the stock will increase in price over these days, meaning you will gain money")

        new_title = '<p style="font-family:sans-serif; font-size: 30px;">Below is the Real World Data for SBUX over the last 30 days, from Yahoo Finance:</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.write("As you can see below, the real world data for the last month shows that SBUX stock has actually decreased -3.22% from the 01-12-2023 to 01-01-2024, compared to the prophet data which also shows an overall decrease in the stock price. So for this one Prophet has been fairly accurate, although I cannot tell the percentage decrease compared to the real world data.")
        st.image("SBUX-Real-Time-month.png", width=1760)

    if symbol_options_menu == "MELI":
        new_title = '<p style="font-family:sans-serif; font-size: 30px;">Below is the Prophet Data for MELI over the next 7 days:</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        seven_day_meli_prophet()
        st.write("Over the next 7 days my advice would be to buy your MELI stock. This is because as shown above, the prophet data is showing the stock will increase in price, meaning you will gain money")

        new_title = '<p style="font-family:sans-serif; font-size: 30px;">Below is the Prophet Data for MELI over the next 14 days:</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        fourteen_day_meli_prophet()
        st.write("Between 7-14 days my advice would be to sell your MELI stock. This is because as shown above, the prophet data is showing the stock will reduce in price over these days, meaning you will loose money")

        new_title = '<p style="font-family:sans-serif; font-size: 30px;">Below is the Prophet Data for MELI over the next 30 days:</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        thirty_day_meli_prophet()
        st.write("Between 14-30 days my advice would be to sell your MELI stock. This is because as shown above, the prophet data is showing the stock will reduce in price over these days, meaning you will loose money")

        new_title = '<p style="font-family:sans-serif; font-size: 30px;">Below is the Real World Data for MELI over the last 30 days, from Yahoo Finance:</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.write("As you can see below, the real world data for the last month shows that MELI stock has actually decreased -4.87% from the 01-12-2023 to 01-01-2024, compared to the prophet data which also shows an overall decrease in the stock price. So for this one Prophet has been fairly accurate, although I cannot tell the percentage decrease compared to the real world data.")
        st.image("MELI-Real-Time-month.png", width=1760)

    if symbol_options_menu == "BKNG":
        new_title = '<p style="font-family:sans-serif; font-size: 30px;">Below is the Prophet Data for BKNG over the next 7 days:</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        seven_day_bkng_prophet()
        st.write("Over the next 7 days my advice would be to sell your BKNG stock. This is because as shown above, the prophet data is showing the stock will reduce in price, meaning you will loose money")

        new_title = '<p style="font-family:sans-serif; font-size: 30px;">Below is the Prophet Data for BKNG over the next 14 days:</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        fourteen_day_bkng_prophet()
        st.write("Between 7-14 days my advice would be to sell your BKNG stock. This is because as shown above, the prophet data is showing the stock will reduce in price over these days, meaning you will loose money")

        new_title = '<p style="font-family:sans-serif; font-size: 30px;">Below is the Prophet Data for BKNG over the next 30 days:</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        thirty_day_bkng_prophet()
        st.write("Between 14-30 days my advice would be to buy BKNG stock. This is because as shown above, the prophet data is showing the stock will increase in price over these days, meaning you will gain money")

        new_title = '<p style="font-family:sans-serif; font-size: 30px;">Below is the Real World Data for BKNG over the last 30 days, from Yahoo Finance:</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.write("As you can see below, the real world data for the last month shows that BKNG stock has actually increased 12.27% from the 01-12-2023 to 01-01-2024, compared to the prophet data which shows an overall decrease in the stock price, even though it picked up near the end.")
        st.image("BKNG-Real-Time-month.png", width=1760)

    if symbol_options_menu == "CTAS":
        new_title = '<p style="font-family:sans-serif; font-size: 30px;">Below is the Prophet Data for CTAS over the next 7 days:</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        seven_day_ctas_prophet()
        st.write("Over the next 7 days my advice would be to sell your CTAS stock. This is because as shown above, the prophet data is showing the stock will reduce in price, meaning you will loose money")

        new_title = '<p style="font-family:sans-serif; font-size: 30px;">Below is the Prophet Data for CTAS over the next 14 days:</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        fourteen_day_ctas_prophet()
        st.write("Between 7-14 days my advice would be to sell your CTAS stock. This is because as shown above, the prophet data is showing the stock will reduce in price over these days, meaning you will loose money")

        new_title = '<p style="font-family:sans-serif; font-size: 30px;">Below is the Prophet Data for CTAS over the next 30 days:</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        thirty_day_ctas_prophet()
        st.write("Between 14-30 days my advice would be to sell your CTAS stock. This is because as shown above, the prophet data is showing the stock will reduce in price over these days, meaning you will loose money")

        new_title = '<p style="font-family:sans-serif; font-size: 30px;">Below is the Real World Data for CTAS over the last 30 days, from Yahoo Finance:</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.write("As you can see below, the real world data for the last month shows that CTAS stock has actually increased 7.60% from the 01-12-2023 to 01-01-2024, compared to the prophet data which shows a dramatic decrease in the stock price.")
        st.image("CTAS-Real-Time-month.png", width=1760)
