import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Union
from datetime import datetime

def load_and_transform_csv(path: str, replace_nans: bool = True) -> Tuple[np.ndarray, Dict[str, int], Dict[str, int], Dict[datetime, int], datetime, datetime, List[str]]:
    """
    Load data from CSV file(s) and transform it into a numpy tensor with log-scaled prices and returns.

    Args:
    - path: Path to the CSV file.
    - replace_nans (bool): Whether to replace NaNs with 0 in the tensor.

    Returns:
    - Tuple containing:
      - tensor (np.ndarray): A 3D numpy array of shape (T, N, J).
      - symbol_to_index (Dict[str, int]): A mapping from symbols to their respective indices.
      - feature_to_index (Dict[str, int]): A mapping from feature names to their respective indices.
      - date_to_index (Dict[datetime, int]): A mapping from dates to their respective indices.
      - start_date (datetime): The start date of the data.
      - end_date (datetime): The end date of the data.
      - features (List[str]): List of feature names.
    """
    df = pd.read_csv(path)
    
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'ticker']]
    
    # Ensure the Date column is datetime type
    df['Date'] = pd.to_datetime(df['Date'])

    # Get unique symbols (tickers) and features
    unique_symbols = df['ticker'].unique()
    features = [col for col in df.columns if col not in ['Date', 'ticker']]

    # Calculate returns and add as a new feature
    df['return'] = df.groupby('ticker')['Close'].diff()
    
    # Get the date range
    start_date = df['Date'].min()
    end_date = df['Date'].max()
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
    df['date_idx'] = df['Date'].map(date_to_index)
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