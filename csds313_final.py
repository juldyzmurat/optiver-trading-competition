#%%
import sys
import os
import collections

# Data Science Libraries
import pandas as pd
import numpy as np
import optuna

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Statistics Libraries
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.graphics.api as smg

# ML Libraries
import sklearn as sk
import lightgbm
import xgboost
import catboost

# Project Libraries

# Configure Visualization
%matplotlib inline
plt.style.use('bmh')

# Configure Pandas and SKLearn
pd.set_option("display.max_colwidth", 20)
pd.set_option("display.precision", 3)
sk.set_config(display="diagram")

# File Specific Configurations
DATA_DIR = "/data/"
Xy_train = pd.read_csv("data/train.csv")
Xy_train
# %%
#count nans 
nan_counts = Xy_train.isnull().sum()

# Displaying the results
print("NaN counts in each column:")
print(nan_counts)

# %%
#drop the far_price and #near price 
Xy_train = Xy_train.drop(['far_price', 'near_price'], axis=1)
Xy_train
# %%
#fill in for missing values
columns_to_impute = ['imbalance_size', 'reference_price', 'matched_size', 'bid_price', 'ask_price', 'wap', 'target']

# Impute missing values for each specified column with the median
for column in columns_to_impute:
    median_value = Xy_train[column].median()
    Xy_train[column] = Xy_train[column].fillna(median_value)


# %%
#mutual information
    
from sklearn.feature_selection import mutual_info_regression

# Xy_train = Xy_train.head(10510)

y_train = Xy_train['target']
X_train = Xy_train.drop(columns=['target', 'row_id'], inplace=False)

mutual_info_scores = mutual_info_regression(X_train, y_train)
feature_mutual_info_scores = pd.Series(mutual_info_scores, index=X_train.columns, name="Mutual_Information")
print("Mutual information between each feature and target:")
print(feature_mutual_info_scores)

#%%
#feature correlation heatmap 
from mlxtend.plotting import heatmap

legal_Xy_train = Xy_train.drop(columns=['row_id'], inplace=False)

correlation_coefs = legal_Xy_train.corr()
corresponding_heatmap = heatmap(correlation_coefs.values, row_names=correlation_coefs.columns, column_names=correlation_coefs.columns)
plt.tight_layout()
plt.show()

# %%
#Create stock-specific features capturing temporal order book dynamics for each stock individually.
#Calculate changes in bid and ask prices, order sizes, spread, and volume.
#Apply time-weighting to the features based on their relevance over time.

Xy_train['numerical_timestamp'] = Xy_train['date_id'] * 1e9 + Xy_train['time_id'] * 1e6 + Xy_train['seconds_in_bucket'] * 1e3

# Sort the DataFrame by 'stock_id' and 'timestamp'
Xy_train = Xy_train.sort_values(by=['stock_id', 'numerical_timestamp'])

# Calculate time differences in seconds for each stock
Xy_train['time_diff'] = Xy_train.groupby('stock_id')['numerical_timestamp'].diff()

# Calculate changes in bid and ask prices
Xy_train['bid_price_change'] = Xy_train.groupby('stock_id')['bid_price'].diff()
Xy_train['ask_price_change'] = Xy_train.groupby('stock_id')['ask_price'].diff()

# Calculate changes in bid and ask sizes
Xy_train['bid_size_change'] = Xy_train.groupby('stock_id')['bid_size'].diff()
Xy_train['ask_size_change'] = Xy_train.groupby('stock_id')['ask_size'].diff()

# Calculate spread
Xy_train['spread'] = Xy_train['ask_price'] - Xy_train['bid_price']

# Calculate volume
Xy_train['volume'] = Xy_train['bid_size'] + Xy_train['ask_size']

# Calculate weighted features based on time differences
time_weight = Xy_train['time_diff'] / Xy_train.groupby('stock_id')['time_diff'].transform('sum')

# Apply time-weighting to the features
Xy_train['bid_price_change_weighted'] = Xy_train['bid_price_change'] * time_weight
Xy_train['ask_price_change_weighted'] = Xy_train['ask_price_change'] * time_weight
Xy_train['bid_size_change_weighted'] = Xy_train['bid_size_change'] * time_weight
Xy_train['ask_size_change_weighted'] = Xy_train['ask_size_change'] * time_weight
Xy_train['spread_weighted'] = Xy_train['spread'] * time_weight
Xy_train['volume_weighted'] = Xy_train['volume'] * time_weight
Xy_train['imbalance_size_target'] = Xy_train.groupby('stock_id')['imbalance_size'].transform(lambda x: x.shift(-1))

#%%
columns_with_missing = Xy_train.columns[Xy_train.isnull().any()].tolist()

# Get the indexes of missing rows for each column
missing_indexes_dict = {}

# Get the indexes of missing rows for each column
for column in columns_with_missing:
    missing_rows = Xy_train[Xy_train[column].isnull()]
    missing_indexes_dict[column] = missing_rows.index

# Combine all the missing row indexes into a single set
all_missing_indexes = set()
for indexes in missing_indexes_dict.values():
    all_missing_indexes.update(indexes)

# Drop the rows with missing values from the DataFrame
Xy_train = Xy_train.drop(index=all_missing_indexes)

# Reset the index after dropping rows
Xy_train = Xy_train.reset_index(drop=True)

# Display the updated DataFrame
print(Xy_train.head())

# Display the updated DataFrame

#%%
#only leave 20% 
Xy_train_sorted = Xy_train.sort_values(by=['stock_id', 'date_id','time_id','seconds_in_bucket'])

# Define a function to leave only the first 20% of data for each stock_id
def leave_first_20_percent(group):
    group_size = len(group)
    return group.head(int(0.2 * group_size))

# Apply the function to leave only the first 20% for each stock_id
Xy_train_first_20_percent = Xy_train_sorted.groupby('stock_id').apply(leave_first_20_percent)

# Reset the index after filtering
Xy_train_first_20_percent.reset_index(drop=True, inplace=True)

#%%
#bismillah
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from tqdm import tqdm

# Assuming Xy_train is your DataFrame with features and target

# Define a function to perform forward selection
def forward_selection(X, y, test_size, early_stopping_rounds=None):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

    features = list(X.columns)
    selected_features = []
    train_errors = []
    val_errors = []

    with tqdm(total=len(features), desc="Forward Selection") as pbar:
        while features:
            best_feature = None
            best_error = float('inf')

            for feature in features:
                candidate_features = selected_features + [feature]

                # Train XGBoost model
                model = xgb.XGBRegressor(objective='reg:squarederror', learning_rate=0.01)
                model.fit(X_train[candidate_features], y_train)

                # Predict and calculate errors
                y_train_pred = model.predict(X_train[candidate_features])
                y_val_pred = model.predict(X_val[candidate_features])

                train_error = mean_squared_error(y_train, y_train_pred)
                val_error = mean_squared_error(y_val, y_val_pred)

                if val_error < best_error:
                    best_feature = feature
                    best_error = val_error

            if early_stopping_rounds is not None and len(val_errors) > early_stopping_rounds and \
                    val_errors[-early_stopping_rounds:] == sorted(val_errors[-early_stopping_rounds:]):
                # Early stopping if validation error does not improve for a certain number of rounds
                break

            selected_features.append(best_feature)
            features.remove(best_feature)
            train_errors.append(train_error)
            val_errors.append(val_error)

            pbar.update(1)

    return selected_features, train_errors, val_errors

# Iterate over different percentages of data
for data_percentage in [1.0, 0.8, 0.6, 0.4, 0.2]:
    print(f"\nForward Selection for {data_percentage * 100}% of data:")

    # Sample data based on percentage
    sampled_data = Xy_train_first_20_percent.sample(frac=data_percentage, random_state=42)

    X_sampled = sampled_data.drop(columns=['target', 'row_id'])
    y_sampled = sampled_data['target']

    selected_features, train_errors, val_errors = forward_selection(X_sampled, y_sampled, test_size=0.2, early_stopping_rounds=5)

    # Display results
    result_df = pd.DataFrame({'Selected Features': selected_features, 'Train MSE': train_errors, 'Validation MSE': val_errors})
    print(result_df)

# %%
#check LOOCV 
from sklearn.model_selection import train_test_split

# Initialize an empty list to store model performance metrics for each stock
fold_metrics = []

# Extract features (X) and target variable (y) for all stocks
X_all = Xy_train.drop(columns=['target', 'row_id'])
y_all = Xy_train['target']

xgb_params = {
    'objective': 'reg:squarederror',  # Regression task
    'eval_metric': 'rmse',  # Root Mean Squared Error
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100
}

# Train the XGBoost model once on the combined dataset
model = xgb.XGBRegressor(**xgb_params)
try:
    model.fit(X_all, y_all)
except Exception as e:
    print(f"Error occurred during model training: {e}")

# Loop over stocks for validation
for i, stock_id in enumerate(Xy_train['stock_id'].unique()):
    print(f"Validating for Stock {stock_id}...")

    # Extract features (X) and target variable (y) for the current stock
    stock_data = Xy_train[Xy_train['stock_id'] == stock_id]
    try:
        # Make predictions on the validation set for the current stock
        y_pred_stock = model.predict(X_val_stock)

        # Calculate RMSE for the current stock
        rmse_stock = mean_squared_error(y_val_stock, y_pred_stock, squared=False)
        print(f"RMSE for Stock {stock_id}: {rmse_stock}\n")

        # Store the metric for this stock
        fold_metrics.append({'stock_id': stock_id, 'rmse': rmse_stock})

    except Exception as e:
        print(f"Error occurred for Stock {stock_id}: {e}")

# Display the overall performance metrics
print("Overall Performance Metrics:")
for fold_metric in fold_metrics:
    print(f"Stock {fold_metric['stock_id']} RMSE: {fold_metric['rmse']}")


# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import time
import xgboost as xgb
import numpy as np


# Load your dataset
# Replace 'your_dataset.csv' with the actual path or DataFrame variable
data = Xy_train

# Assuming 'target' is your target variable
X = data.drop(['target'], axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Filter-based feature selection
start_time = time.time()
selector = SelectKBest(f_regression, k=5)  # Adjust k as needed
X_train_filtered = selector.fit_transform(X_train, y_train)
X_test_filtered = selector.transform(X_test)
end_time = time.time()
print(f"Filter-based feature selection time: {end_time - start_time} seconds")

# Wrapper feature selection (RFE with Linear Regression)
start_time = time.time()
estimator = LinearRegression()
selector = RFE(estimator, n_features_to_select=5)  # Adjust the number of features as needed
X_train_wrapper = selector.fit_transform(X_train, y_train)
X_test_wrapper = selector.transform(X_test)
end_time = time.time()
print(f"Wrapper feature selection time: {end_time - start_time} seconds")

# Embedded feature selection (Random Forest)
# start_time = time.time()
# estimator = RandomForestRegressor(n_estimators=100, random_state=42)
# estimator.fit(X_train, y_train)
# importances = estimator.feature_importances_
# feature_indices = (-importances).argsort()[:5]  # Adjust the number of features as needed
# X_train_embedded = X_train.iloc[:, feature_indices]
# X_test_embedded = X_test.iloc[:, feature_indices]
# end_time = time.time()
# print(f"Embedded feature selection time: {end_time - start_time} seconds")

# Train models and evaluate
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_elements = y_true != 0
    return np.mean(np.abs((y_true[nonzero_elements] - y_pred[nonzero_elements]) / y_true[nonzero_elements])) * 100

def evaluate_model(X_train, X_test, y_train, y_test):
    model = xgb.XGBRegressor(learning_rate=0.001, n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    ssr = ((y_pred - y_test.mean()) ** 2).sum()
    sst = ((y_test - y_test.mean()) ** 2).sum()
    sse = ((y_test - y_pred) ** 2).sum()

    print(f"R-squared: {r2}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    print(f"Sum of Squares Regression: {ssr}")
    print(f"Sum of Squares Total: {sst}")
    print(f"Sum of Squares Error: {sse}")


# Evaluate filter-based model
print("\nFilter-based model evaluation:")
evaluate_model(X_train_filtered, X_test_filtered, y_train, y_test)

# Evaluate wrapper model
print("\nWrapper model evaluation:")
evaluate_model(X_train_wrapper, X_test_wrapper, y_train, y_test)

# Evaluate embedded model
# print("\nEmbedded model evaluation:")
# evaluate_model(X_train_embedded, X_test_embedded, y_train, y_test)


#%%
#revisit the three feature selection methods from math perspective 
#compare lstm with xgboost - be careful with the format
#test presence and absence of wap 
#%%