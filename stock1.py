from pathlib import Path
import yfinance as yf
import pandas as pd
import numpy as np
from IPython.display import display
from datetime import date
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from talib import abstract
from tqdm import tqdm
from sklearn.preprocessing import minmax_scale, maxabs_scale
import pickle
import os
from stock_funcs import *

dir_path = os.path.dirname(os.path.realpath(__file__))

full_data = pd.read_csv(Path(dir_path,"full_data.csv"), header=[0,1], index_col=[0])
full_data = full_data.sort_index()
empty_cols = [col for col in full_data.columns if full_data[col].isnull().all()]
full_data.drop(empty_cols, axis=1, inplace=True)

basic_indicators = [s for s in full_data.columns.get_level_values(0).unique().tolist()]
nasdaq_tickers_df = pd.read_csv(Path(dir_path,"nasdaqlisted.txt"), sep="|")
nasdaq_tickers = [t for t in nasdaq_tickers_df["Symbol"][:-1].tolist() if not pd.isnull(t)]

slice_data, target_data, dates_data = get_all_sliced_data(full_data, nasdaq_tickers)

with open('stock_slice_data.pkl', 'wb') as outp:
    pickle.dump(slice_data, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(target_data, outp, pickle.HIGHEST_PROTOCOL)
    pickle.dump(dates_data, outp, pickle.HIGHEST_PROTOCOL)
