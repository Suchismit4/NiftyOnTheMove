import datetime
import logging
import numpy as np
from typing import Dict, Optional
import argparse

from src import Strato
from src.struct.strategy import Strategy
from src.struct.indicator import Indicator, IndicatorCalculator
from src.flow.position import Position
from src.utils import load_and_transform_csv

class MovingAverage(Indicator):
    def __init__(self, window: int):
        self.window = window

    def init(self, data: np.ndarray) -> np.ndarray:
        if data.shape[0] < self.window:
            return np.full((1,), np.nan)
        
        data_slice_sum = data[:self.window].sum(axis=0)
        moving_average = (data_slice_sum / self.window)
        return (moving_average, self.window)

    def step(self, current_value: np.ndarray, new_data: np.ndarray, previous_result: np.ndarray) -> np.ndarray:
        return previous_result + (new_data - current_value) / self.window

class MovingAverageStrategy(Strategy):
   
    def __init__(self, short_window: int, long_window: int):
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window
        self.crossover_state: Dict[str, Optional[str]] = {}
        self.initialized: Dict[str, bool] = {}

    def generate_signals(self, date_idx: int, indicator_calculator: IndicatorCalculator,
                         positions: Dict[str, Position], symbol_to_index: Dict[str, int]):
        if date_idx < self.long_window:
            return  # Not enough data to generate signals

        short_ma = indicator_calculator.get_indicator(f'MA_{self.short_window}')
        long_ma = indicator_calculator.get_indicator(f'MA_{self.long_window}')
        new_crossover_state = dict(self.crossover_state)
        new_signals = {}

        for symbol, symbol_idx in symbol_to_index.items():
            short_mavg = short_ma[date_idx, symbol_idx]
            long_mavg = long_ma[date_idx, symbol_idx]

            if np.isnan(short_mavg) or np.isnan(long_mavg):
                continue

            if symbol not in self.initialized:
                self.initialized[symbol] = False

            current_state = new_crossover_state.get(symbol)

            if not self.initialized[symbol]:
                # Initialize crossover state without generating signals
                if short_mavg > long_mavg:
                    new_crossover_state[symbol] = 'bullish'
                elif short_mavg < long_mavg:
                    new_crossover_state[symbol] = 'bearish'
                self.initialized[symbol] = True
                new_signals[symbol] = self.HOLD  # No trading signal during initialization
            else:
                # Generate trading signals based on crossover
                if short_mavg > long_mavg and current_state != 'bullish':
                    new_signals[symbol] = self.BUY
                    new_crossover_state[symbol] = 'bullish'
                elif short_mavg < long_mavg and current_state != 'bearish':
                    new_signals[symbol] = self.SELL
                    new_crossover_state[symbol] = 'bearish'
                else:
                    new_signals[symbol] = self.HOLD

        self.crossover_state = new_crossover_state
        self.signals = new_signals

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
        
    data, symbol_to_index, feature_to_index, date_to_index, start_date, end_date, features = load_and_transform_csv(csv_paths)
    # Call the setup_logging function at the start of your main script
    if args.verbose: 
        setup_logging()
    # Create strategy
    strategy = MovingAverageStrategy(short_window=10, long_window=30)

    # Import benchmark
    import pandas as pd
    bchk = pd.read_csv('NIFTY500.csv')

    # Create Strato instance
    strato = Strato(data,
                    symbol_to_index,
                    feature_to_index,
                    date_to_index,
                    starting_cash=100000.0,
                    trade_size=10,
                    strategy=strategy, benchmark=bchk)

    # Add indicators
    strato.add_indicator('MA_10', MovingAverage(10))
    strato.add_indicator('MA_30', MovingAverage(30))

    # Run backtest
    results = strato.run_backtest()

    print(f'Starting Portfolio Value: ₹{100000.00}')
    print(f'Final Portfolio Value: ₹{results[-1]:.2f}')
    
