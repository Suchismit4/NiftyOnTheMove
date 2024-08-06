import numpy as np
from typing import Dict, Optional

from ..struct.strategy import Strategy
from ..struct.indicator import Indicator, IndicatorCalculator
from ..flow.position import Position

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
        self.name = "Moving Average Cross Strategy (Long Only) Simple"
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