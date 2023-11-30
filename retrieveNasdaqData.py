# Import libraries
import yfinance as yf
import pandas as pd
from colorama import Fore


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
