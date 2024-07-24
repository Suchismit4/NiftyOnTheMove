import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(path):
    """
    Load data from a CSV file.

    Args:
    path (str): Path to the CSV file.

    Returns:
    pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    all_data = pd.read_csv(path)
    return all_data

def roll_symbols_check(df):
    """
    Check the number of unique symbols for each date in the DataFrame.
    Log when the number is more than 500 or less than 500.

    Args:
    df (pd.DataFrame): Input DataFrame containing 'Date' and 'Stock' columns.

    Returns:
    pd.Series: A series with dates as index and the count of unique symbols for each date.
    """
    # Ensure 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Count unique symbols for each date
    symbol_counts = df.groupby('Date')['Stock'].nunique()

    # Log dates where the count is not equal to 500
    for date, count in symbol_counts.items():
        if count != 500:
            if count > 500:
                logging.warning(f"Date {date.date()}: {count} symbols (more than 500)")
            else:
                logging.warning(f"Date {date.date()}: {count} symbols (less than 500)")

    return symbol_counts

def main():
    # Load the data
    data_path = './final_dataset.csv'  
    df = load_data(data_path)

    # Check symbol counts
    symbol_counts = roll_symbols_check(df)

    # Additional summary statistics
    logging.info(f"Total number of unique dates: {len(symbol_counts)}")
    logging.info(f"Average number of symbols per date: {symbol_counts.mean():.2f}")
    logging.info(f"Minimum number of symbols on a single date: {symbol_counts.min()}")
    logging.info(f"Maximum number of symbols on a single date: {symbol_counts.max()}")

if __name__ == "__main__":
    main()