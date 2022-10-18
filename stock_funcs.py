import pandas as pd
import numpy as np
from talib import abstract
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models

# region helper functions

def normalize(ts:pd.Series) -> pd.Series:
    """
    Normalize the time series. If the series includes negative numbers normalizes to [-1,1]. Otherwise normalizes to [0,1].
    """
    copy = ts.copy()
    min_ = max(0,min(copy))
    copy -= min_
    max_ = max(abs(copy))
    copy /= max_
    return copy

def normalize_mult(ts_list):
    if not isinstance(ts_list,list):
        ts_list = [ts_list]
    norm_ts_list = []
    for ts in ts_list:
        norm_ts_list.append(pd.Series(normalize(ts), name=ts.name))
    return norm_ts_list

def clean(ts: pd.Series) -> pd.Series:
    """
    Clean the time series. Remove nans in the beginning and fill in missing values
    """
    copy = ts.copy()
    i1 = copy.first_valid_index()
    copy = copy[i1:].interpolate()
    return copy

def visualize_stock_mult(stock_series, t=None, figsize=(15,10)):
    if not isinstance(stock_series,list):
        stock_series = [stock_series]
    plt.figure(figsize=figsize)
    for series in stock_series:
        plt.plot(series, label=series.name)
    title = t if t is not None else ""
    plt.title(title, fontsize=16)

    # Define the labels for x-axis and y-axis
    plt.ylabel('Price', fontsize=14)
    plt.xlabel('Year', fontsize=14)

    # Plot the grid lines
    plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
    plt.legend()

    # Show the plot
    plt.show()

# endregion

# region technical functions

def get_technical_indicators(ts_data):
    """
    Returns a list of indicator series
    ts_data should be a df that includes CHLOV
    """
    technical_indicators_series = []
    column_labels = []

    # Weighted moving average
    technical_indicators_series.append(abstract.WMA(ts_data))
    column_labels.append("wma")

    # Moving Average Convergence Divergence 
    macd = abstract.MACD(ts_data)
    # technical_indicators_series.extend([macd['macd'], macd['macdsignal'], macd["macdhist"]])
    # column_labels.extend(['macd', 'macdsignal', 'macdhist'])
    technical_indicators_series.append(macd['macdhist'])
    column_labels.append('macdhist')

    # Relative Strength Index
    technical_indicators_series.append(abstract.RSI(ts_data))
    column_labels.append("rsi")

    # Channel Commodity Index
    technical_indicators_series.append(abstract.CCI(ts_data))
    column_labels.append('cci')

    # Stochastic Indicator - take the difference
    stoch = abstract.STOCH(ts_data)
    technical_indicators_series.append(stoch['slowk'] - stoch['slowd'])
    column_labels.append('stochdif')

    # Bollinger Bands
    bbands = abstract.BBANDS(ts_data)
    technical_indicators_series.extend([bbands['upperband'], bbands['middleband'], bbands["lowerband"]])
    column_labels.extend(['bbupper', 'bbmiddle', 'bblower'])

    # William %R
    technical_indicators_series.append(-abstract.WILLR(ts_data))
    column_labels.append('willr')

    # Average Directional Index
    technical_indicators_series.append(abstract.ADX(ts_data))
    column_labels.append("adx")

    # On Balance Volume
    technical_indicators_series.append(abstract.OBV(ts_data))
    column_labels.append("obv")

    # Money Flow Index
    technical_indicators_series.append(abstract.RSI(ts_data))
    column_labels.append('mfi')

    technical_indicators_series = pd.DataFrame(technical_indicators_series).T
    technical_indicators_series.columns = column_labels

    return technical_indicators_series

def get_percent_changes(full_data, ticker, standardize, index_fund, lookahead = 1):
    diff = full_data['Close'][ticker].diff(lookahead)[lookahead:].tolist()
    perc = diff/full_data['Close'][ticker][:-lookahead]
    if standardize:
        index_diff = index_fund["Close"].diff(lookahead)[lookahead:]
        index_perc = index_diff/index_fund["Close"][:-lookahead].tolist()
        perc = index_perc - perc
    perc.name = 'perc_change'
    return perc

def get_ohlc_data(full_data, ticker, index_fund, basic_indicators=['Close', 'High', 'Low', 'Open', 'Volume'], standardize = False):
    ticker_data = []
    for i in basic_indicators:
        ind = full_data[i]
        
        if ticker not in ind:
            return None

        d = clean(ind[ticker])
        if standardize and not i=="Volume":
            snp_ind = clean(index_fund[i])
            d = clean(d/snp_ind)

        d = pd.Series(d, name=i.lower())
        ticker_data.append(d)
    return ticker_data

# endregion 

# region data functions

def get_data_for_ticker(ticker, full_data, index_fund, standardize = False):
    ticker_data = get_ohlc_data(full_data, ticker, index_fund, standardize=standardize)
    if ticker_data is None:
        return None
    ticker_data = pd.DataFrame(ticker_data).T
    ticker_data = pd.concat([ticker_data, get_technical_indicators(ticker_data)], axis=1)
    ticker_data = pd.concat([ticker_data, get_percent_changes(full_data, ticker, standardize, index_fund)], axis=1)
    return ticker_data

def create_slices_for_ticker(ticker, full_data, index_fund, k = 20, lookahead = 0, standardize = True):
    indicators = get_data_for_ticker(ticker, full_data, index_fund, standardize=standardize)
    if indicators is None:
        return None
    indicators = indicators.dropna()
    slices = []
    targets = []
    dates = []
    for i in range(k, len(indicators)):
        slice = indicators.iloc[i-k: i]
        slice_norm = pd.DataFrame(normalize_mult([slice[col] for col in slice])).T
        slice_norm.index = slice.index
        slices.append(slice_norm)

        pred = indicators.iloc[i]
        targets.append(pred['perc_change'])
        dates.append(pred.name)
    return slices, targets, dates

def get_all_sliced_data(data, tickers, index_fund = None):
    standardize = index_fund is not None

    slices = []
    targets = []
    dates = []
    for ticker in tqdm(tickers):
        res = create_slices_for_ticker(ticker,data, index_fund, standardize=standardize)
        if res is None:
            continue
        s,t,d = res
        slices.extend(s)
        targets.extend(t)
        dates.extend(d)
    return slices, targets, dates

def create_train_test_data(slices, targets, dates, classification = True):
    slices = np.asarray(slices)
    targets = np.array(targets)
    targets_classification = targets>0
    dates = np.array(dates)
    cutoff = '2022-01-01'
    date_inds = dates > cutoff
    test_inds = date_inds
    train_inds = np.logical_not(date_inds)
    X_train, X_test = slices[train_inds], slices[test_inds]
    if classification:
        y_train, y_test = targets_classification[train_inds], targets_classification[test_inds]
    else:
        y_train, y_test = targets[train_inds], targets[test_inds]
    return (X_train, X_test, y_train, y_test)

# endregion

# region model functions

def evaluate_model(X_train, X_test, y_train, y_test):
    # start with a simpler model - 1 layer
    # epochs - maybe get to 100, save every 10 epochs
    # try leaving it as regression

    model = models.Sequential()
    model.add(layers.Conv1D(16, 4, activation='relu', input_shape=(20, 18)))
    model.add(layers.Dropout(0.5)) # check order of dropout
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(32, 3, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation="sigmoid"))
    # Binary classification 
    model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=15, 
                        validation_data=(X_test, y_test))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
    return model, test_acc

# endregion
