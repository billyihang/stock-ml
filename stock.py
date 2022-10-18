import yfinance as yf
import pandas as pd
import numpy as np
from IPython.display import display
from datetime import date
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from talib import abstract
from stock_funcs import *


dir_path = Path(__file__).parent.resolve()
stock_csv_path = Path(dir_path, "full_data.csv")

"""
NASDAQ makes this information available via FTP and they update it every night. Log into ftp.nasdaqtrader.com anonymously. 
Look in the directory SymbolDirectory. You'll notice two files: nasdaqlisted.txt and otherlisted.txt. These two files will give you 
the entire list of tradeable symbols, where they are listed, their name/description, and an indicator as to whether they are an ETF.
"""

nasdaq_tickers_df = pd.read_csv("nasdaqlisted.txt", sep="|")
nasdaq_tickers = [t for t in nasdaq_tickers_df["Symbol"][:-1].tolist() if not pd.isnull(t)]
len(nasdaq_tickers)

# Set the start and end date
start_date = '1990-01-01'
end_date = str(date.today())

# get index fund: snps
snp = yf.download('SNP', start_date, end_date)
snp.index = [str(d)[:10] for d in snp.index]

if stock_csv_path == None:
    data = yf.download(nasdaq_tickers,start_date, end_date, auto_adjust=True, threads=False)
    data = data.sort_index()[:-1]
    empty_cols = [col for col in data.columns if data[col].isnull().all()]
    data.drop(empty_cols, axis=1, inplace=True)
else:
    data = pd.read_csv("full_data.csv", header=[0,1], index_col=[0])
    data.index = [s[:10] for s in data.index]
    basic_indicators = [s for s in data.columns.get_level_values(0).unique().tolist()]

print(data.head())


