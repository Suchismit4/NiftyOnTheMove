import numpy as np
from typing import Dict, Tuple

from ..struct.strategy import Strategy
from ..struct.indicator import Indicator, IndicatorCalculator
from ..flow.position import Position
import matplotlib.pyplot as plt

class RSI(Indicator):
    def __init__(self, window: int, inverse_logistic: bool = True):
        self.window = window
        self.inverse_logistic = inverse_logistic

    def init(self, data: np.ndarray) -> Tuple[np.ndarray, int]:
        if data.shape[0] < self.window + 1:
            return np.full((3, data.shape[1]), np.nan), 0
        
        close = data
        upsum = np.full(data.shape[1], 1e-60)
        dnsum = np.full(data.shape[1], 1e-60)
        
        for i in range(1, self.window):
            diff = close[i] - close[i-1]
            upsum += np.where(diff > 0, diff, 0)
            dnsum -= np.where(diff < 0, diff, 0)
        
        upsum /= (self.window - 1)
        dnsum /= (self.window - 1)
        
        diff = close[self.window] - close[self.window - 1]
        upsum = np.where(diff > 0,
                         ((self.window - 1.) * upsum + diff) / self.window,
                         upsum * (self.window - 1.) / self.window)
        dnsum = np.where(diff < 0,
                         ((self.window - 1.) * dnsum - diff) / self.window,
                         dnsum * (self.window - 1.) / self.window)
        
        rsi = 100 * upsum / (upsum + dnsum)
        
        if self.inverse_logistic:
            rsi = -10. * np.log(2. / (1 + 0.00999 * (2 * rsi - 100)) - 1)
        
        return np.stack((upsum, dnsum)), rsi, self.window

    def step(self, current_value: np.ndarray, new_data: np.ndarray, carry: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        upsum, dnsum = carry
        diff = new_data - current_value
        upsum = np.where(diff > 0,
                         ((self.window - 1.) * upsum + diff) / self.window,
                         upsum * (self.window - 1.) / self.window)
        dnsum = np.where(diff < 0,
                         ((self.window - 1.) * dnsum - diff) / self.window,
                         dnsum * (self.window - 1.) / self.window)
        rsi = 100 * upsum / (upsum + dnsum)
        if self.inverse_logistic:
            rsi = -10. * np.log(2. / (1 + 0.00999 * (2 * rsi - 100)) - 1)
        return np.vstack((upsum, dnsum)), rsi
    
class OutOfWhackMarketAnomoly(Strategy): 
    
    def __init__(self, rsi_short_window: int, rsi_long_window: int, threshold: float):
        super().__init__()
        self.name = f"Out Of Whack RSI {rsi_short_window}-{rsi_long_window} regression (Long)"
        self.short_window = rsi_short_window
        self.long_window = rsi_long_window
        self.threshold = threshold
        self.initialized: Dict[str, bool] = {}
        self.positions: Dict[str, int] = {}
        self.detrend_length = 200
        
    def _detrend_rsi_single(self, short_rsi: np.array, long_rsi: np.array):
        length = self.detrend_length
        if len(short_rsi) < length:
            return np.nan

        xmean = np.mean(long_rsi[-length:])
        ymean = np.mean(short_rsi[-length:])

        xdiff = long_rsi[-length:] - xmean
        ydiff = short_rsi[-length:] - ymean

        xss = np.sum(xdiff * xdiff)
        xy = np.sum(xdiff * ydiff)

        coef = xy / (xss + 1e-60)

        return ydiff[-1] - coef * xdiff[-1]
        
    def generate_signals(self, date_idx: int, indicator_calculator: IndicatorCalculator,
                         positions: Dict[str, Position], symbol_to_index: Dict[str, int]):
        if date_idx < self.detrend_length:
            return  # Not enough data to generate signals

        short_rsi = indicator_calculator.get_indicator(f'RSI_{self.short_window}')
        long_rsi = indicator_calculator.get_indicator(f'RSI_{self.long_window}')

        new_signals = {}

        for symbol, symbol_idx in symbol_to_index.items():
            if symbol not in self.initialized:
                self.initialized[symbol] = False
                self.positions[symbol] = 0

            if not self.initialized[symbol]:
                # Initialize without generating signals
                self.initialized[symbol] = True
                new_signals[symbol] = self.HOLD
                continue

            current_short_rsi = short_rsi[:date_idx+1, symbol_idx]
            current_long_rsi = long_rsi[:date_idx+1, symbol_idx]

            current_detrended_rsi = self._detrend_rsi_single(current_short_rsi, current_long_rsi)

            if np.isnan(current_detrended_rsi):
                new_signals[symbol] = self.HOLD
                continue

            if current_detrended_rsi < -self.threshold and self.positions[symbol] == 0:
                new_signals[symbol] = self.BUY
                self.positions[symbol] = 1
            elif self.positions[symbol] == 1:
                if positions[symbol].bars_since_entry >= 1:
                    new_signals[symbol] = self.SELL
                    self.positions[symbol] = 0
                else:
                    new_signals[symbol] = self.HOLD
            else:
                new_signals[symbol] = self.HOLD

        self.signals = new_signals
