import pandas as pd
import datetime
import argparse
import logging

from src import Strato
from src.utils import load_and_transform_csv

# from src.examples.moving_average_cross import *
from src.examples.out_of_whack import *

def setup_logging():
    import os
    # Create a unique directory for logs
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a unique log file name based on the current time
    log_file = os.path.join(log_dir, f"log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Configure logging to log to the file
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

# Main execution
if __name__ == "__main__":
    
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Main script for backtesting")
    parser.add_argument("csv", help="The backtest data file paths", type=str, nargs='+')
    parser.add_argument("--verbose", help="Dump detailed log", action='store_true')
    args = parser.parse_args()

    # Access the list of CSV paths
    csv_paths = args.csv

    # Print out the list of CSV paths (for debugging purposes)
    if args.verbose:
        print(f"CSV paths provided: {csv_paths}")
        
    # Import benchmark
    bchk = pd.read_csv('NIFTY500.csv')
    bchk['ticker'] = 'NIFTY500'
    bchk.to_csv('NIFTY500_fixed.csv')        

    data, symbol_to_index, feature_to_index, date_to_index, start_date, end_date, features = load_and_transform_csv(csv_paths)
    # Call the setup_logging function at the start of your main script
    if args.verbose: 
        setup_logging()
        
        
    # Create strategy
    strategy = OutOfWhackMarketAnomoly(rsi_short_window=2, 
                                       rsi_long_window=25,
                                       threshold=15)

    # Create a Strato instance for a strategy
    strato = Strato(data,
                    symbol_to_index,
                    feature_to_index,
                    date_to_index,
                    starting_cash=100000.0,
                    trade_size=5,
                    strategy=strategy, benchmark=bchk, generate_report=True)

    # Add indicators
    strato.add_indicator('RSI_2', RSI(2, inverse_logistic=True))
    strato.add_indicator('RSI_25', RSI(25))

    # Run backtest
    results = strato.run_backtest()

    print(f'Starting Portfolio Value: ₹{100000.00}')
    print(f'Final Portfolio Value: ₹{results[-1]:.2f}')
    
        
    # # Create strategy
    # strategy = MovingAverageStrategy(short_window=10, long_window=30)

    # # Create a Strato instance for a strategy
    # strato = Strato(data,
    #                 symbol_to_index,
    #                 feature_to_index,
    #                 date_to_index,
    #                 starting_cash=100000.0,
    #                 trade_size=10,
    #                 strategy=strategy, benchmark=bchk)

    # # Add indicators
    # strato.add_indicator('MA_10', MovingAverage(10))
    # strato.add_indicator('MA_30', MovingAverage(30))

    # # Run backtest
    # results = strato.run_backtest()

    # print(f'Starting Portfolio Value: ₹{100000.00}')
    # print(f'Final Portfolio Value: ₹{results[-1]:.2f}')
    
    
