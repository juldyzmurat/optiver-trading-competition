#%%
import os
import time

import pandas as pd
import numpy as np

from numba import njit, prange

import gc

import lightgbm as lgbm

from sklearn.metrics import mean_absolute_error

import warnings 
from itertools import combinations  
from warnings import simplefilter
# %%
# 📊 Define flags and variables
warnings.filterwarnings("ignore")
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
is_offline = False 
is_train = True  
is_infer = True 
max_lookback = np.nan 
split_day = 435
SEED = 42
# %%
lgbm_params = {
    "objective": "mae",
    "n_estimators": 6000,
    "num_leaves": 256,
    "subsample": 0.6,
    'learning_rate': 0.00871, 
    'max_depth': 11,
    "colsample_bytree" : 0.8,
    "n_jobs": 4,
    "device": "gpu",
    "verbosity": -1,
    "importance_type": "gain",
    "random_state": SEED,
    "max_bin": 247
}
# %%
# 📂 Read the dataset from a CSV file using Pandas
df = pd.read_csv("data/train.csv")
df = df[df['stock_id'].between(0, 9)]
df

# 🧹 Remove rows with missing values in the "target" column
df = df.dropna(subset=["target"])

# 🔁 Reset the index of the DataFrame and apply the changes in place
df.reset_index(drop=True, inplace=True)

# 📏 Get the shape of the DataFrame (number of rows and columns)
df_shape = df.shape
df_shape
# %%
def reduce_memory_usage(df):
    print("Memory Usage Before Optimization:")
    print(df.memory_usage(deep=True).sum() / (1024 ** 2), "MB")

    # 🔄 Iterate through each column in the DataFrame
    for col in df.columns:
        col_type = df[col].dtype

        # Check if the column's data type is not 'object' (i.e., numeric)
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Check if the column's data type is an integer
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # Check if the column's data type is a float
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)
                    
    # Display the memory usage after optimization
    print("\nMemory Usage After Optimization:")
    print(df.memory_usage(deep=True).sum() / (1024 ** 2), "MB")
    return df

# %%
@njit(parallel=True)
def compute_triplet_imbalance(df_values, comb_indices):
    num_rows = df_values.shape[0]
    num_combinations = len(comb_indices)
    imbalance_features = np.empty((num_rows, num_combinations))
    for i in prange(num_combinations):
        a, b, c = comb_indices[i]
        for j in range(num_rows):
            max_val = max(df_values[j, a], df_values[j, b], df_values[j, c])
            min_val = min(df_values[j, a], df_values[j, b], df_values[j, c])
            mid_val = df_values[j, a] + df_values[j, b] + df_values[j, c] - min_val - max_val
            
            if mid_val == min_val:
                imbalance_features[j, i] = np.nan
            else:
                imbalance_features[j, i] = (max_val - mid_val) / (mid_val - min_val)

    return imbalance_features

def calculate_triplet_imbalance_numba(price, df):
    df_values = df[price].values
    comb_indices = [(price.index(a), price.index(b), price.index(c)) for a, b, c in combinations(price, 3)]
    features_array = compute_triplet_imbalance(df_values, comb_indices)
    columns = [f"{a}_{b}_{c}_imb2" for a, b, c in combinations(price, 3)]
    features = pd.DataFrame(features_array, columns=columns)
    return features
# %%
@njit(fastmath=True)
def rolling_average(arr, window):
    """
    Calculate the rolling average for a 1D numpy array.
    
    Parameters:
    arr (numpy.ndarray): Input array to calculate the rolling average.
    window (int): The number of elements to consider for the moving average.
    
    Returns:
    numpy.ndarray: Array containing the rolling average values.
    """
    n = len(arr)
    result = np.empty(n)
    result[:window] = np.nan  # Padding with NaN for elements where the window is not full
    cumsum = np.cumsum(arr)

    for i in range(window, n):
        result[i] = (cumsum[i] - cumsum[i - window]) / window

    return result

@njit(parallel=True)
def compute_rolling_averages(df_values, window_sizes):
    """
    Calculate the rolling averages for multiple window sizes in parallel.
    
    Parameters:
    df_values (numpy.ndarray): 2D array of values to calculate the rolling averages.
    window_sizes (List[int]): List of window sizes for the rolling averages.
    
    Returns:
    numpy.ndarray: A 3D array containing the rolling averages for each window size.
    """
    num_rows, num_features = df_values.shape
    num_windows = len(window_sizes)
    rolling_features = np.empty((num_rows, num_features, num_windows))

    for feature_idx in prange(num_features):
        for window_idx, window in enumerate(window_sizes):
            rolling_features[:, feature_idx, window_idx] = rolling_average(df_values[:, feature_idx], window)

    return rolling_features
# %%
def imbalance_features(df):
    # Define lists of price and size-related column names
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]
    
    # V1 features
    # Calculate various features using Pandas eval function
    df["volume"] = df.eval("ask_size + bid_size")
    df["mid_price"] = df.eval("(ask_price + bid_price) / 2")
    df["liquidity_imbalance"] = df.eval("(bid_size-ask_size)/(bid_size+ask_size)")
    df["matched_imbalance"] = df.eval("(imbalance_size-matched_size)/(matched_size+imbalance_size)")
    df["size_imbalance"] = df.eval("bid_size / ask_size")
    
    # Create features for pairwise price imbalances
    for c in combinations(prices, 2):
        df[f"{c[0]}_{c[1]}_imb"] = df.eval(f"({c[0]} - {c[1]})/({c[0]} + {c[1]})")
    
    # Calculate triplet imbalance features using the Numba-optimized function
    for c in [['ask_price', 'bid_price', 'wap', 'reference_price'], sizes]:
        triplet_feature = calculate_triplet_imbalance_numba(c, df)
        df[triplet_feature.columns] = triplet_feature.values
    
    # V2 features
    # Calculate additional features   
    df["imbalance_momentum"] = df.groupby(['stock_id'])['imbalance_size'].diff(periods=1) / df['matched_size']
    df["price_spread"] = df["ask_price"] - df["bid_price"]
    df["spread_intensity"] = df.groupby(['stock_id'])['price_spread'].diff()
    df['price_pressure'] = df['imbalance_size'] * (df['ask_price'] - df['bid_price'])
    df['market_urgency'] = df['price_spread'] * df['liquidity_imbalance']
    df['depth_pressure'] = (df['ask_size'] - df['bid_size']) * (df['far_price'] - df['near_price'])
    
    # Calculate various statistical aggregation features
    for func in ["mean", "std", "skew", "kurt"]:
        df[f"all_prices_{func}"] = df[prices].agg(func, axis=1)
        df[f"all_sizes_{func}"] = df[sizes].agg(func, axis=1)
        
    # V3 features
    # Calculate shifted and return features for specific columns
    for col in ['matched_size', 'imbalance_size', 'reference_price', 'imbalance_buy_sell_flag']:
        for window in [1, 2, 3, 10]:
            df[f"{col}_shift_{window}"] = df.groupby('stock_id')[col].shift(window)
            df[f"{col}_ret_{window}"] = df.groupby('stock_id')[col].pct_change(window)
    
    # V4 features - Rolling averages
    window_sizes = [1, 2, 3, 8, 10]  # Define your desired window sizes
    for price in prices:
        rolling_avg_features = compute_rolling_averages(df[price].values.reshape(-1, 1), window_sizes)

        # Assigning the rolling average results to the DataFrame
        for i, window in enumerate(window_sizes):
            column_name = f"{price}_rolling_avg_{window}"
            df[column_name] = rolling_avg_features[:, 0, i]
    
    # Calculate diff features for specific columns
    for col in ['ask_price', 'bid_price', 'ask_size', 'bid_size', 'market_urgency', 'imbalance_momentum', 'size_imbalance']:
        for window in [1, 2, 3, 10]:
            df[f"{col}_diff_{window}"] = df.groupby("stock_id")[col].diff(window)

    return df.replace([np.inf, -np.inf], 0)

def other_features(df):
    df["dow"] = df["date_id"] % 5  # Day of the week
    df["seconds"] = df["seconds_in_bucket"] % 60  
    df["minute"] = df["seconds_in_bucket"] // 60  
    for key, value in global_stock_id_feats.items():
        df[f"global_{key}"] = df["stock_id"].map(value.to_dict())

    return df
# %%
@njit(parallel = True)
def calculate_rsi(prices, period=14):
    rsi_values = np.zeros_like(prices)

    for col in prange(prices.shape[1]):
        price_data = prices[:, col]
        delta = np.zeros_like(price_data)
        delta[1:] = price_data[1:] - price_data[:-1]
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = np.mean(gain[:period])
        avg_loss = np.mean(loss[:period])
        
        if avg_loss != 0:
            rs = avg_gain / avg_loss
        else:
            rs = 1e-9  # or any other appropriate default value
            
        rsi_values[:period, col] = 100 - (100 / (1 + rs))

        for i in prange(period-1, len(price_data)-1):
            avg_gain = (avg_gain * (period - 1) + gain[i]) / period
            avg_loss = (avg_loss * (period - 1) + loss[i]) / period
            if avg_loss != 0:
                rs = avg_gain / avg_loss
            else:
                rs = 1e-9  # or any other appropriate default value
            rsi_values[i+1, col] = 100 - (100 / (1 + rs))

    return rsi_values
# %%
@njit(parallel=True)
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    rows, cols = data.shape
    macd_values = np.empty((rows, cols))
    signal_line_values = np.empty((rows, cols))
    histogram_values = np.empty((rows, cols))

    for i in prange(cols):
        short_ema = np.zeros(rows)
        long_ema = np.zeros(rows)

        for j in range(1, rows):
            short_ema[j] = (data[j, i] - short_ema[j - 1]) * (2 / (short_window + 1)) + short_ema[j - 1]
            long_ema[j] = (data[j, i] - long_ema[j - 1]) * (2 / (long_window + 1)) + long_ema[j - 1]

        macd_values[:, i] = short_ema - long_ema

        signal_line = np.zeros(rows)
        for j in range(1, rows):
            signal_line[j] = (macd_values[j, i] - signal_line[j - 1]) * (2 / (signal_window + 1)) + signal_line[j - 1]

        signal_line_values[:, i] = signal_line
        histogram_values[:, i] = macd_values[:, i] - signal_line

    return macd_values, signal_line_values, histogram_values
# %%
@njit(parallel=True)
def calculate_bband(data, window=20, num_std_dev=2):
    num_rows, num_cols = data.shape
    upper_bands = np.zeros_like(data)
    lower_bands = np.zeros_like(data)
    mid_bands = np.zeros_like(data)

    for col in prange(num_cols):
        for i in prange(window - 1, num_rows):
            window_slice = data[i - window + 1 : i + 1, col]
            mid_bands[i, col] = np.mean(window_slice)
            std_dev = np.std(window_slice)
            upper_bands[i, col] = mid_bands[i, col] + num_std_dev * std_dev
            lower_bands[i, col] = mid_bands[i, col] - num_std_dev * std_dev

    return upper_bands, mid_bands, lower_bands
# %%
def generate_ta(df):
    # Define lists of price and size-related column names
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
    # sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]
    
    for stock_id, values in df.groupby(['stock_id'])[prices]:
        # RSI
        col_rsi = [f'rsi_{col}' for col in values.columns]
        rsi_values = calculate_rsi(values.values)
        df.loc[values.index, col_rsi] = rsi_values
        gc.collect()
        
        # MACD
        macd_values, signal_line_values, histogram_values = calculate_macd(values.values)
        col_macd = [f'macd_{col}' for col in values.columns]
        col_signal = [f'macd_sig_{col}' for col in values.columns]
        col_hist = [f'macd_hist_{col}' for col in values.columns]
        
        df.loc[values.index, col_macd] = macd_values
        df.loc[values.index, col_signal] = signal_line_values
        df.loc[values.index, col_hist] = histogram_values
        gc.collect()
        
        # Bollinger Bands
        bband_upper_values, bband_mid_values, bband_lower_values = calculate_bband(values.values, window=20, num_std_dev=2)
        col_bband_upper = [f'bband_upper_{col}' for col in values.columns]
        col_bband_mid = [f'bband_mid_{col}' for col in values.columns]
        col_bband_lower = [f'bband_lower_{col}' for col in values.columns]
        
        df.loc[values.index, col_bband_upper] = bband_upper_values
        df.loc[values.index, col_bband_mid] = bband_mid_values
        df.loc[values.index, col_bband_lower] = bband_lower_values
        gc.collect()
    
    return df
# %%
def generate_all_features(df):
    cols = [c for c in df.columns if c not in ["row_id", "time_id", "target"]]
    df = df[cols]
    
    # Generate TA features
    df = generate_ta(df)
    gc.collect() # Perform garbage collection to free up memory
    
    # Generate imbalance features
    df = imbalance_features(df)
    df = other_features(df)
    gc.collect()  
    
    feature_names = [c for c in df.columns if c not in ["row_id", "target", "time_id", "date_id"]]
    
    return df[feature_names]
# %%
if is_offline:
    # In offline mode, split the data into training and validation sets based on the split_day
    df_train = df[df["date_id"] <= split_day]
    df_valid = df[df["date_id"] > split_day]
    
    # Display a message indicating offline mode and the shapes of the training and validation sets
    print("Offline mode")
    print(f"train : {df_train.shape}, valid : {df_valid.shape}")
else:
    # In online mode, use the entire dataset for training
    df_train = df
    
    # Display a message indicating online mode
    print("Online mode")
# %%
if is_train:
    global_stock_id_feats = {
        "median_size": df_train.groupby("stock_id")["bid_size"].median() + df_train.groupby("stock_id")["ask_size"].median(),
        "std_size": df_train.groupby("stock_id")["bid_size"].std() + df_train.groupby("stock_id")["ask_size"].std(),
        "ptp_size": df_train.groupby("stock_id")["bid_size"].max() - df_train.groupby("stock_id")["bid_size"].min(),
        "median_price": df_train.groupby("stock_id")["bid_price"].median() + df_train.groupby("stock_id")["ask_price"].median(),
        "std_price": df_train.groupby("stock_id")["bid_price"].std() + df_train.groupby("stock_id")["ask_price"].std(),
        "ptp_price": df_train.groupby("stock_id")["bid_price"].max() - df_train.groupby("stock_id")["ask_price"].min(),
    }
    if is_offline:
        df_train_feats = generate_all_features(df_train)
        print("Build Train Feats Finished.")
        df_valid_feats = generate_all_features(df_valid)
        print("Build Valid Feats Finished.")
        df_valid_feats = reduce_mem_usage(df_valid_feats)
    else:
        df_train_feats = generate_all_features(df_train)
        print("Build Online Train Feats Finished.")

    df_train_feats = reduce_memory_usage(df_train_feats)
# %%
@njit(parallel=True)
def compute_triplet_imbalance(df_values, comb_indices):
    num_rows = df_values.shape[0]
    num_combinations = len(comb_indices)
    imbalance_features = np.empty((num_rows, num_combinations))
    for i in prange(num_combinations):
        a, b, c = comb_indices[i]
        for j in range(num_rows):
            max_val = max(df_values[j, a], df_values[j, b], df_values[j, c])
            min_val = min(df_values[j, a], df_values[j, b], df_values[j, c])
            mid_val = df_values[j, a] + df_values[j, b] + df_values[j, c] - min_val - max_val
            
            if mid_val == min_val:
                imbalance_features[j, i] = np.nan
            else:
                imbalance_features[j, i] = (max_val - mid_val) / (mid_val - min_val)

    return imbalance_features

def calculate_triplet_imbalance_numba(price, df):
    df_values = df[price].values
    comb_indices = [(price.index(a), price.index(b), price.index(c)) for a, b, c in combinations(price, 3)]
    features_array = compute_triplet_imbalance(df_values, comb_indices)
    columns = [f"{a}_{b}_{c}_imb2" for a, b, c in combinations(price, 3)]
    features = pd.DataFrame(features_array, columns=columns)
    return features
# %%
@njit(fastmath=True)
def rolling_average(arr, window):
    """
    Calculate the rolling average for a 1D numpy array.
    
    Parameters:
    arr (numpy.ndarray): Input array to calculate the rolling average.
    window (int): The number of elements to consider for the moving average.
    
    Returns:
    numpy.ndarray: Array containing the rolling average values.
    """
    n = len(arr)
    result = np.empty(n)
    result[:window] = np.nan  # Padding with NaN for elements where the window is not full
    cumsum = np.cumsum(arr)

    for i in range(window, n):
        result[i] = (cumsum[i] - cumsum[i - window]) / window

    return result

@njit(parallel=True)
def compute_rolling_averages(df_values, window_sizes):
    """
    Calculate the rolling averages for multiple window sizes in parallel.
    
    Parameters:
    df_values (numpy.ndarray): 2D array of values to calculate the rolling averages.
    window_sizes (List[int]): List of window sizes for the rolling averages.
    
    Returns:
    numpy.ndarray: A 3D array containing the rolling averages for each window size.
    """
    num_rows, num_features = df_values.shape
    num_windows = len(window_sizes)
    rolling_features = np.empty((num_rows, num_features, num_windows))

    for feature_idx in prange(num_features):
        for window_idx, window in enumerate(window_sizes):
            rolling_features[:, feature_idx, window_idx] = rolling_average(df_values[:, feature_idx], window)

    return rolling_features
# %%
def imbalance_features(df):
    # Define lists of price and size-related column names
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]
    
    # V1 features
    # Calculate various features using Pandas eval function
    df["volume"] = df.eval("ask_size + bid_size")
    df["mid_price"] = df.eval("(ask_price + bid_price) / 2")
    df["liquidity_imbalance"] = df.eval("(bid_size-ask_size)/(bid_size+ask_size)")
    df["matched_imbalance"] = df.eval("(imbalance_size-matched_size)/(matched_size+imbalance_size)")
    df["size_imbalance"] = df.eval("bid_size / ask_size")
    
    # Create features for pairwise price imbalances
    for c in combinations(prices, 2):
        df[f"{c[0]}_{c[1]}_imb"] = df.eval(f"({c[0]} - {c[1]})/({c[0]} + {c[1]})")
    
    # Calculate triplet imbalance features using the Numba-optimized function
    for c in [['ask_price', 'bid_price', 'wap', 'reference_price'], sizes]:
        triplet_feature = calculate_triplet_imbalance_numba(c, df)
        df[triplet_feature.columns] = triplet_feature.values
    
    # V2 features
    # Calculate additional features   
    df["imbalance_momentum"] = df.groupby(['stock_id'])['imbalance_size'].diff(periods=1) / df['matched_size']
    df["price_spread"] = df["ask_price"] - df["bid_price"]
    df["spread_intensity"] = df.groupby(['stock_id'])['price_spread'].diff()
    df['price_pressure'] = df['imbalance_size'] * (df['ask_price'] - df['bid_price'])
    df['market_urgency'] = df['price_spread'] * df['liquidity_imbalance']
    df['depth_pressure'] = (df['ask_size'] - df['bid_size']) * (df['far_price'] - df['near_price'])
    
    # Calculate various statistical aggregation features
    for func in ["mean", "std", "skew", "kurt"]:
        df[f"all_prices_{func}"] = df[prices].agg(func, axis=1)
        df[f"all_sizes_{func}"] = df[sizes].agg(func, axis=1)
        
    # V3 features
    # Calculate shifted and return features for specific columns
    for col in ['matched_size', 'imbalance_size', 'reference_price', 'imbalance_buy_sell_flag']:
        for window in [1, 2, 3, 10]:
            df[f"{col}_shift_{window}"] = df.groupby('stock_id')[col].shift(window)
            df[f"{col}_ret_{window}"] = df.groupby('stock_id')[col].pct_change(window)
    
    # V4 features - Rolling averages
    window_sizes = [1, 2, 3, 8, 10]  # Define your desired window sizes
    for price in prices:
        rolling_avg_features = compute_rolling_averages(df[price].values.reshape(-1, 1), window_sizes)

        # Assigning the rolling average results to the DataFrame
        for i, window in enumerate(window_sizes):
            column_name = f"{price}_rolling_avg_{window}"
            df[column_name] = rolling_avg_features[:, 0, i]
    
    # Calculate diff features for specific columns
    for col in ['ask_price', 'bid_price', 'ask_size', 'bid_size', 'market_urgency', 'imbalance_momentum', 'size_imbalance']:
        for window in [1, 2, 3, 10]:
            df[f"{col}_diff_{window}"] = df.groupby("stock_id")[col].diff(window)

    return df.replace([np.inf, -np.inf], 0)

def other_features(df):
    df["dow"] = df["date_id"] % 5  # Day of the week
    df["seconds"] = df["seconds_in_bucket"] % 60  
    df["minute"] = df["seconds_in_bucket"] // 60  
    for key, value in global_stock_id_feats.items():
        df[f"global_{key}"] = df["stock_id"].map(value.to_dict())

    return df
# %%
@njit(parallel = True)
def calculate_rsi(prices, period=14):
    rsi_values = np.zeros_like(prices)

    for col in prange(prices.shape[1]):
        price_data = prices[:, col]
        delta = np.zeros_like(price_data)
        delta[1:] = price_data[1:] - price_data[:-1]
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = np.mean(gain[:period])
        avg_loss = np.mean(loss[:period])
        
        if avg_loss != 0:
            rs = avg_gain / avg_loss
        else:
            rs = 1e-9  # or any other appropriate default value
            
        rsi_values[:period, col] = 100 - (100 / (1 + rs))

        for i in prange(period-1, len(price_data)-1):
            avg_gain = (avg_gain * (period - 1) + gain[i]) / period
            avg_loss = (avg_loss * (period - 1) + loss[i]) / period
            if avg_loss != 0:
                rs = avg_gain / avg_loss
            else:
                rs = 1e-9  # or any other appropriate default value
            rsi_values[i+1, col] = 100 - (100 / (1 + rs))

    return rsi_values
# %%
@njit(parallel=True)
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    rows, cols = data.shape
    macd_values = np.empty((rows, cols))
    signal_line_values = np.empty((rows, cols))
    histogram_values = np.empty((rows, cols))

    for i in prange(cols):
        short_ema = np.zeros(rows)
        long_ema = np.zeros(rows)

        for j in range(1, rows):
            short_ema[j] = (data[j, i] - short_ema[j - 1]) * (2 / (short_window + 1)) + short_ema[j - 1]
            long_ema[j] = (data[j, i] - long_ema[j - 1]) * (2 / (long_window + 1)) + long_ema[j - 1]

        macd_values[:, i] = short_ema - long_ema

        signal_line = np.zeros(rows)
        for j in range(1, rows):
            signal_line[j] = (macd_values[j, i] - signal_line[j - 1]) * (2 / (signal_window + 1)) + signal_line[j - 1]

        signal_line_values[:, i] = signal_line
        histogram_values[:, i] = macd_values[:, i] - signal_line

    return macd_values, signal_line_values, histogram_values
# %%
@njit(parallel=True)
def calculate_bband(data, window=20, num_std_dev=2):
    num_rows, num_cols = data.shape
    upper_bands = np.zeros_like(data)
    lower_bands = np.zeros_like(data)
    mid_bands = np.zeros_like(data)

    for col in prange(num_cols):
        for i in prange(window - 1, num_rows):
            window_slice = data[i - window + 1 : i + 1, col]
            mid_bands[i, col] = np.mean(window_slice)
            std_dev = np.std(window_slice)
            upper_bands[i, col] = mid_bands[i, col] + num_std_dev * std_dev
            lower_bands[i, col] = mid_bands[i, col] - num_std_dev * std_dev

    return upper_bands, mid_bands, lower_bands
# %%
def generate_ta(df):
    # Define lists of price and size-related column names
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
    # sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]
    
    for stock_id, values in df.groupby(['stock_id'])[prices]:
        # RSI
        col_rsi = [f'rsi_{col}' for col in values.columns]
        rsi_values = calculate_rsi(values.values)
        df.loc[values.index, col_rsi] = rsi_values
        gc.collect()
        
        # MACD
        macd_values, signal_line_values, histogram_values = calculate_macd(values.values)
        col_macd = [f'macd_{col}' for col in values.columns]
        col_signal = [f'macd_sig_{col}' for col in values.columns]
        col_hist = [f'macd_hist_{col}' for col in values.columns]
        
        df.loc[values.index, col_macd] = macd_values
        df.loc[values.index, col_signal] = signal_line_values
        df.loc[values.index, col_hist] = histogram_values
        gc.collect()
        
        # Bollinger Bands
        bband_upper_values, bband_mid_values, bband_lower_values = calculate_bband(values.values, window=20, num_std_dev=2)
        col_bband_upper = [f'bband_upper_{col}' for col in values.columns]
        col_bband_mid = [f'bband_mid_{col}' for col in values.columns]
        col_bband_lower = [f'bband_lower_{col}' for col in values.columns]
        
        df.loc[values.index, col_bband_upper] = bband_upper_values
        df.loc[values.index, col_bband_mid] = bband_mid_values
        df.loc[values.index, col_bband_lower] = bband_lower_values
        gc.collect()
    
    return df
# %%
def generate_all_features(df):
    cols = [c for c in df.columns if c not in ["row_id", "time_id", "target"]]
    df = df[cols]
    
    # Generate TA features
    df = generate_ta(df)
    gc.collect() # Perform garbage collection to free up memory
    
    # Generate imbalance features
    df = imbalance_features(df)
    df = other_features(df)
    gc.collect()  
    
    feature_names = [c for c in df.columns if c not in ["row_id", "target", "time_id", "date_id"]]
    
    return df[feature_names]
# %%
# Check if the code is running in offline or online mode
if is_offline:
    # In offline mode, split the data into training and validation sets based on the split_day
    df_train = df[df["date_id"] <= split_day]
    df_valid = df[df["date_id"] > split_day]
    
    # Display a message indicating offline mode and the shapes of the training and validation sets
    print("Offline mode")
    print(f"train : {df_train.shape}, valid : {df_valid.shape}")
else:
    # In online mode, use the entire dataset for training
    df_train = df
    
    # Display a message indicating online mode
    print("Online mode")
# %%
if is_train:
    global_stock_id_feats = {
        "median_size": df_train.groupby("stock_id")["bid_size"].median() + df_train.groupby("stock_id")["ask_size"].median(),
        "std_size": df_train.groupby("stock_id")["bid_size"].std() + df_train.groupby("stock_id")["ask_size"].std(),
        "ptp_size": df_train.groupby("stock_id")["bid_size"].max() - df_train.groupby("stock_id")["bid_size"].min(),
        "median_price": df_train.groupby("stock_id")["bid_price"].median() + df_train.groupby("stock_id")["ask_price"].median(),
        "std_price": df_train.groupby("stock_id")["bid_price"].std() + df_train.groupby("stock_id")["ask_price"].std(),
        "ptp_price": df_train.groupby("stock_id")["bid_price"].max() - df_train.groupby("stock_id")["ask_price"].min(),
    }
    if is_offline:
        df_train_feats = generate_all_features(df_train)
        print("Build Train Feats Finished.")
        df_valid_feats = generate_all_features(df_valid)
        print("Build Valid Feats Finished.")
        df_valid_feats = reduce_mem_usage(df_valid_feats)
    else:
        df_train_feats = generate_all_features(df_train)
        print("Build Online Train Feats Finished.")

    df_train_feats = reduce_memory_usage(df_train_feats)
# %%
feature_name = list(df_train_feats.columns)
print(f"Feature length = {len(feature_name)}")

# The total number of date_ids is 480, we split them into 5 folds with a gap of 5 days in between
num_folds = 5
fold_size = 480 // num_folds
gap = 5

models = []
scores = []

model_save_path = 'artifacts'  # Directory to save models
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# We need to use the date_id from df_train to split the data
date_ids = df_train['date_id'].values

for i in range(num_folds):
    start = i * fold_size
    end = start + fold_size
    
    # Define the training and testing sets by date_id
    if i < num_folds - 1:  # No need to purge after the last fold
        purged_start = end - 2
        purged_end = end + gap + 2
        train_indices = (date_ids >= start) & (date_ids < purged_start) | (date_ids > purged_end)
    else:
        train_indices = (date_ids >= start) & (date_ids < end)
    
    test_indices = (date_ids >= end) & (date_ids < end + fold_size)
    
    df_fold_train = df_train_feats[train_indices]
    df_fold_train_target = df_train['target'][train_indices]
    df_fold_valid = df_train_feats[test_indices]
    df_fold_valid_target = df_train['target'][test_indices]

    print(f"Fold {i+1} Model Training")
    
    # Train a LightGBM model for the current fold
    lgbm_model = lgbm.LGBMRegressor(**lgbm_params)
    lgbm_model.fit(
        df_fold_train[feature_name],
        df_fold_train_target,
        eval_set=[(df_fold_valid[feature_name], df_fold_valid_target)],
        callbacks=[
            lgbm.callback.early_stopping(stopping_rounds=100),
            lgbm.callback.log_evaluation(period=100),
        ],
    )

    # Append the model to the list
    models.append(lgbm_model)
    # Save the model to a file
    model_filename = os.path.join(model_save_path, f'model_{i+1}.txt')
    lgbm_model.booster_.save_model(model_filename)
    print(f"Model for fold {i+1} saved to {model_filename}")

    # Evaluate model performance on the validation set
    fold_predictions = lgbm_model.predict(df_fold_valid[feature_name])
    fold_score = mean_absolute_error(fold_predictions, df_fold_valid_target)
    scores.append(fold_score)
    print(f"Fold {i+1} MAE: {fold_score}")

    # Free up memory by deleting fold specific variables
    del df_fold_train, df_fold_train_target, df_fold_valid, df_fold_valid_target
    gc.collect()

# Calculate the average best iteration from all regular folds
average_best_iteration = int(np.mean([model.best_iteration_ for model in models]))

# Update the lgbm_params with the average best iteration
final_model_params = lgbm_params.copy()
final_model_params['n_estimators'] = average_best_iteration

print(f"Training final model with average best iteration: {average_best_iteration}")

# Train the final model on the entire dataset
final_model = lgbm.LGBMRegressor(**final_model_params)
final_model.fit(
    df_train_feats[feature_name],
    df_train['target'],
    callbacks=[
        lgbm.callback.log_evaluation(period=100),
    ],
)

# Append the final model to the list of models
models.append(final_model)

# Save the final model to a file
final_model_filename = os.path.join(model_save_path, 'final_model.txt')
final_model.booster_.save_model(final_model_filename)
print(f"Final model saved to {final_model_filename}")

# Now 'models' holds the trained models for each fold and 'scores' holds the validation scores
print(f"Average MAE across all folds: {np.mean(scores)}")
#%%
lgbm.plot_importance(lgbm_model, max_num_features=25, importance_type="gain", grid=False, precision=1)
#%%
def zero_sum(prices, volumes):
    std_error = np.sqrt(volumes)
    step = np.sum(prices) / np.sum(std_error)
    out = prices - std_error * step
    return out

if is_infer:
    import optiver2023
    env = optiver2023.make_env()
    iter_test = env.iter_test()
    counter = 0
    y_min, y_max = -64, 64
    qps, predictions = [], []
    cache = pd.DataFrame()

    # Weights for each fold model
    model_weights = [1/len(models)] * len(models) 
    
    for (test, revealed_targets, sample_prediction) in iter_test:
        now_time = time.time()
        cache = pd.concat([cache, test], ignore_index=True, axis=0)
        
        if counter > 0:
            cache = cache.groupby(['stock_id']).tail(21).sort_values(by=['date_id', 'seconds_in_bucket', 'stock_id']).reset_index(drop=True)
        feat = generate_all_features(cache)[-len(test):]

        # Generate predictions for each model and calculate the weighted average
        lgbm_predictions = np.zeros(len(test))
        for model, weight in zip(models, model_weights):
            lgbm_predictions += weight * model.predict(feat)

        lgbm_predictions = zero_sum(lgbm_predictions, test['bid_size'] + test['ask_size'])
        clipped_predictions = np.clip(lgbm_predictions, y_min, y_max)
        sample_prediction['target'] = clipped_predictions
        env.predict(sample_prediction)
        counter += 1
        qps.append(time.time() - now_time)
        if counter % 10 == 0:
            print(counter, 'qps:', np.mean(qps))

    time_cost = 1.146 * np.mean(qps)
    print(f"The code will take approximately {np.round(time_cost, 4)} hours to reason about")