import pandas as pd
import datetime
import argparse
import logging

from strato.src import Strato
from strato.src.utils import load_and_transform_csv

from strategy import StocksOnTheMoveByAndrewsClenow
from indicators import *

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
    
if __name__ == "__main__":
    print("Now backtesting NiftyOnTheMove...")
        
    # Import benchmark
    bchk = pd.read_csv('./data/NIFTY500.csv')
    bchk['ticker'] = 'NIFTY500'
    
    # Import benchmark
    constituents = pd.read_csv('./data/NIFTY500_2010_2020.csv')
    
    path = "./data/NIFTY500_2010_2020_HISTORICAL.csv"

    data, symbol_to_index, feature_to_index, date_to_index, start_date, end_date, features = load_and_transform_csv(path)
    # Call the setup_logging function at the start of your main script
    
    setup_logging()
        
    # Create strategy
    strategy = StocksOnTheMoveByAndrewsClenow (
                 constituents    = constituents, 
                 symbol_to_index = symbol_to_index,
                 date_to_index   = date_to_index, 
                 lookback      = 125, 
                 sma_short     = 100,
                 sma_long      = 200, 
                 volatility    = 25,
                 portfolio_at_risk = 0.0015,
                 min_momentum      =  40.,
                 max_stocks        = 20
                 )

    # Create a Strato instance for a strategy
    strato = Strato(data,
                    symbol_to_index,
                    feature_to_index,
                    date_to_index,
                    starting_cash=100000.0,
                    strategy=strategy, benchmark=bchk, generate_report=True)

    # Add indicators
    strato.add_indicator('Momentum_125', Momentum(125, inverse_logistic=False))
    strato.add_indicator('SMA_100', ExponentialMovingAverage(100))
    strato.add_indicator('SMA_200', ExponentialMovingAverage(200))
    strato.add_indicator('ATR_25', AverageTrueRange(25))
    strato.add_indicator('Volatility_25', Volatility(window = 25, annualize= True, ewma=False, parkinson=False))
    strato.add_indicator('Close', Close())


    # Run backtest
    results = strato.run_backtest()

    print(f'Starting Portfolio Value: ₹1,00,000')
    print(f'Final Portfolio Value: ₹{results[-1]:.2f}')
    
    
