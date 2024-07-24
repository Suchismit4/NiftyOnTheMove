import pandas as pd
import os
from tqdm import tqdm
import logging
import numpy as np

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    """
    Load the initial datasets from CSV files.
    
    This function reads two CSV files:
    1. NIFTY500_2010_2020.csv: Contains timeline information for NIFTY500 stocks.
    2. stocks_df.csv: Contains historical price data for stocks.
    
    Returns:
    tuple: A tuple containing two pandas DataFrames (timeline_df, stocks_df)
    """
    timeline_df = pd.read_csv('./data/NIFTY500_2010_2020.csv')
    stocks_df = pd.read_csv('./data/stocks_df.csv')
    logging.info("Data loaded...")
    return timeline_df, stocks_df

def extract(timeline_df: pd.DataFrame, stocks_df: pd.DataFrame):
    unique_stock_symbols = timeline_df['ticker'].unique()
    data = []
    data_to_symbol = {sym: -1 for sym in unique_stock_symbols}
    start_date = pd.to_datetime('2010-01-01')
    end_date = pd.to_datetime('2020-12-31')
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    def process_stock_data(stock_data, reference_data=None):
        # Drop rows with NaN values
        stock_data.dropna(inplace=True)
        
        # Drop duplicate dates
        stock_data = stock_data[~stock_data.index.duplicated(keep='first')]
        
        # Ensure there are valid dates to process
        if stock_data.empty:
            return None

        # Create a full date range including weekends for the entire stock data period
        full_date_range = pd.date_range(start=stock_data.index.min(), end=stock_data.index.max(), freq='D')

        # Reindex the stock data to include all dates
        full_data_with_weekends = stock_data.reindex(full_date_range)

        # Forward fill all initial NaN values from the first valid observation
        full_data_with_weekends.ffill(inplace=True)

        # Forward fill the weekends (Saturday and Sunday) from the previous Fridays
        weekends = (full_data_with_weekends.index.weekday >= 5)
        full_data_with_weekends.loc[weekends] = full_data_with_weekends.ffill().loc[weekends]

        # If reference data is provided, use it to fill missing values
        if reference_data is not None:
            full_data_with_weekends = reference_data.combine_first(full_data_with_weekends)

        # Fill other missing values with 0
        full_data_with_weekends = full_data_with_weekends.fillna(0)

        # Clip the data to the specified date range
        clipped_data = full_data_with_weekends.reindex(date_range).fillna(0)

        return clipped_data.values

    # Read CSV file names from ./Datasets/SCRIP/
    csv_files = [f.split('.')[0] for f in os.listdir('./Datasets/SCRIP') if f.endswith('.csv')]
    
    # Check which symbols are available in the SCRIP dataset
    available_in_scrip = set(unique_stock_symbols) & set(csv_files)
    
    for sym in tqdm(available_in_scrip, desc=f"Processing from SCRIP dataset {len(available_in_scrip)} out of {len(unique_stock_symbols)}"):
        file_path = f'./Datasets/SCRIP/{sym}.csv'
        stock_data = pd.read_csv(file_path)
        stock_data = stock_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data.set_index('Date', inplace=True)
        processed_data = process_stock_data(stock_data)
        if processed_data is not None:
            data.append(processed_data)
            data_to_symbol[sym] = len(data) - 1
    
    # Group the stocks_df by 'Stock' once, outside the loop
    grouped_stocks = stocks_df.groupby('Stock')
    
    for sym in tqdm(unique_stock_symbols, desc="Processing with additional stocks data"):
        if sym in grouped_stocks.groups:
            stock_data = grouped_stocks.get_group(sym)[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            stock_data.set_index('Date', inplace=True)
            
            if data_to_symbol[sym] != -1:
                # If we already have data from SCRIP, use it as the reference
                reference_data = pd.DataFrame(data[data_to_symbol[sym]], index=date_range, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
                stock_data_reindexed = stock_data.reindex(date_range)
                
                # Forward fill all initial NaN values from the first valid observation in stock_data
                stock_data_reindexed.ffill(inplace=True)
                
                # Forward fill the weekends (Saturday and Sunday) from the previous Fridays
                weekends = (stock_data_reindexed.index.weekday >= 5)
                stock_data_reindexed.loc[weekends] = stock_data_reindexed.ffill().loc[weekends]
                
                # Combine the reference data with the new stock data, filling missing values in reference_data
                combined_data = reference_data.combine_first(stock_data_reindexed)
                
                # Replace the existing data with the combined data
                data[data_to_symbol[sym]] = combined_data.values
            else:
                # If we don't have data from SCRIP, process and add the data
                processed_data = process_stock_data(stock_data)
                if processed_data is not None:
                    data.append(processed_data)
                    data_to_symbol[sym] = len(data) - 1

    
    data = np.array(data)
    
    # Print symbols not found in either dataset
    not_found_symbols = [sym for sym, idx in data_to_symbol.items() if idx == -1]
    print("Symbols not found in either dataset:")
    print(not_found_symbols)
    
    return data, data_to_symbol, date_range, not_found_symbols
    
def merge_data(data, data_to_symbol, date_range, timeline_df):
    merged_data = []
    missing_data_symbols = set()
    
    # Convert timeline_df['Date'] to datetime if it's not already
    timeline_df['Event Date'] = pd.to_datetime(timeline_df['Event Date'])
    
    # Group the timeline by date
    timeline_grouped = timeline_df.groupby('Event Date')

    for date in tqdm(date_range, desc="Merging data"):
        # Get the tickers for this date
        if date in timeline_grouped.groups:
            day_tickers = timeline_grouped.get_group(date)['ticker'].tolist()
        else:
            continue  # Skip dates not in the timeline

        for ticker in day_tickers:
            if ticker in data_to_symbol and data_to_symbol[ticker] != -1:
                idx = data_to_symbol[ticker]
                date_idx = (date - date_range[0]).days

                if idx < len(data) and date_idx < len(data[idx]):
                    stock_data = data[idx][date_idx]
                    
                    # Check for missing data
                    if np.all(stock_data == 0):
                        missing_data_symbols.add(ticker)
                    
                    merged_data.append([
                        'Nifty 500',
                        date,
                        stock_data[0],  # Open
                        stock_data[1],  # High
                        stock_data[2],  # Low
                        stock_data[3],  # Close
                        stock_data[4],  # Volume
                        ticker
                    ])

    # Create the final DataFrame
    columns = ['Index Name', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'ticker']
    final_df = pd.DataFrame(merged_data, columns=columns)

    # Convert numeric columns
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    final_df[numeric_columns] = final_df[numeric_columns].astype(float)

    # Sort the DataFrame by Date and ticker
    final_df = final_df.sort_values(['Date', 'ticker'])

    # Reset the index
    final_df.reset_index(drop=True, inplace=True)
    
    # Report missing data symbols
    print("Symbols with missing observations:")
    print(missing_data_symbols)
    
    return final_df
    
timeline, stocks = load_data()
ext, map, dr, not_found_symbols = extract(timeline, stocks)
# Use this function after print(not_found)
final_df = merge_data(ext, map, dr, timeline)
final_df.to_csv('NIFTY500_2010_2020_HISTORICAL.csv')
