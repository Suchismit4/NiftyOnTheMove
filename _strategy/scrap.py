import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
file_path = './data/ICICIBANK.csv'
file_path2 = './data/TCS.csv'

df = pd.read_csv(file_path, parse_dates=['date'])
df2 = pd.read_csv(file_path2, parse_dates=['date'])

df['ticker'] = 'ICICIBANK'
df2['ticker'] = 'TCS'

_df = pd.concat([df, df2])

def load_and_transform_csv(df: pd.DataFrame, replace_nans: bool = True):
    df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']]
    
    # Ensure the Date column is datetime type
    df['date'] = pd.to_datetime(df['date'])

    # Get unique symbols (tickers) and features
    unique_symbols = df['ticker'].unique()
    features = [col for col in df.columns if col not in ['date', 'ticker']]

    # Calculate returns and add as a new feature
    df['return'] = df.groupby('ticker')['close'].diff()
    
    # Get the date range
    start_date = df['date'].min()
    end_date = df['date'].max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Create the tensor shape (T, N, J+1) to include the return feature
    T = len(date_range)
    N = len(unique_symbols)
    J = len(features) + 1  # +1 for the return feature

    # Initialize the tensor with NaNs (to differentiate between actual 0 values and missing data)
    tensor = np.full((T, N, J), np.nan, dtype=np.float64)

    # Create mappings
    symbol_to_index = {symbol: idx for idx, symbol in enumerate(unique_symbols)}
    feature_to_index = {feature: idx for idx, feature in enumerate(features)}
    feature_to_index['return'] = len(features)  # Add the return feature index
    date_to_index = {date: idx for idx, date in enumerate(date_range)}

    # Map indices for efficient tensor population
    df['date_idx'] = df['date'].map(date_to_index)
    df['symbol_idx'] = df['ticker'].map(symbol_to_index)
    date_indices = df['date_idx'].values
    symbol_indices = df['symbol_idx'].values
    feature_values = df[features].values
    return_values = df['return'].values.reshape(-1, 1)
    feature_values = np.hstack((feature_values, return_values))

    # Populate the tensor using numpy indexing
    tensor[date_indices, symbol_indices] = feature_values

    # Replace infinities with NaNs (handle 0/0 cases)
    tensor[np.isinf(tensor)] = np.nan

    # Replace NaNs with 0 for missing values if replace_nans is True
    if replace_nans:
        tensor = np.nan_to_num(tensor)
        
    return tensor, symbol_to_index, feature_to_index, date_to_index, start_date, end_date, list(features) + ['return']

# Assuming _df is your dataframe input
tensor, symbol_to_index, feature_to_index, date_to_index, start_date, end_date, features = load_and_transform_csv(_df)

# Initialize variables
n = 90
x = np.arange(n).reshape(-1, 1)
x = np.tile(x, (1, tensor.shape[1]))

# Create a 2D array to store momentum values for all stocks
manual_mom = np.zeros((tensor.shape[0], tensor.shape[1]))

# Compute xmean and x deviations only once
xmean = np.mean(x, axis=0)
x_dev = x - xmean

for i in range(n, tensor.shape[0]):
    y = np.log(tensor[i - n:i, :, feature_to_index['close']])
    ymean = np.mean(y, axis=0)
    
    # Compute deviations from mean
    y_dev = y - ymean
    
    # Compute covariance components
    ssxm = np.sum(x_dev ** 2, axis=0)
    ssxym = np.sum(x_dev * y_dev, axis=0)
    ssym = np.sum(y_dev ** 2, axis=0)
    
    # Compute correlation coefficient
    r = ssxym / np.sqrt(ssxm * ssym)
    r = np.clip(r, -1, 1)
    
    # Compute slope
    slope = ssxym / ssxm
    
    # Annualize the slope
    annualized = (np.exp(slope * 252)) - 1
    
    # Calculate momentum
    manual_mom[i, :] = (annualized * (r ** 2))

# Plot the histograms of momentum values
plt.figure(figsize=(12, 6))
plt.plot(manual_mom[:, 0], label='Manual Method 0')
plt.plot(manual_mom[:, 1], label='Manual Method 1')
plt.legend(loc='upper right')
plt.xlabel('Index')
plt.ylabel('Momentum')
plt.title('Comparison of Momentum Calculation Methods')
plt.show()