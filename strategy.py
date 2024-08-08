import sys
from strato.src.struct.strategy import Strategy
from strato.src.struct.indicator import IndicatorCalculator
from strato.src.flow.position import Position
from typing import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class StocksOnTheMoveByAndrewsClenow(Strategy): 
    
    def __init__(self,
                 constituents: pd.DataFrame, 
                 symbol_to_index: Dict[str, int],
                 date_to_index: Dict[str, int], 
                 lookback: int = 90, 
                 sma_short: int = 100,
                 sma_long: int = 200, 
                 volatility: int = 20, 
                 _min_momentum: float = 0.,
                 _max_stocks: int = 25
                 ):
        super().__init__()
        self.name = f"Clenow Momentum Strategy (Long only)"
        self.short_window = sma_short
        self.long_window = sma_long
        self._min_momentum = _min_momentum
        self.volatility = volatility
        self.lookback = lookback
        self.max_stocks = _max_stocks
        self.constituents = constituents
        self.index_historical = None
        # Build a mapping for constituents with date and symbol indices
        self.constituents['Event Date'] = pd.to_datetime(self.constituents['Event Date'])

        # Create a dictionary mapping tickers to their indices
        ticker_to_index = {sym: symbol_to_index.get(sym, -1) for sym in self.constituents['ticker'].unique()}

        # Group by 'Event Date' and aggregate the tickers
        grouped = self.constituents.groupby('Event Date')['ticker'].apply(list)

        # Create the final dictionary
        self.const_d2i = {
            date: {sym: ticker_to_index[sym] for sym in tickers if ticker_to_index[sym] != -1}
            for date, tickers in grouped.items()
        }

        self.date_to_index = date_to_index
        
        self.initialized: Dict[str, bool] = {}
        self.positions: Dict[str, Position] = {}
        print("Initialized.")
        
    def _make_rankings(self, close: np.ndarray, mom: np.ndarray, sma_short, date_idx):
        
        qualified_stocks = []
        actual_date = next(date for date, idx in self.date_to_index.items() if idx == date_idx)
        
        for symbol, pos in self.positions.items():
            if symbol in list(self.const_d2i[actual_date].keys()):
                symbol_idx = self.const_d2i[actual_date][symbol]
                
                price_history = close[date_idx-self.lookback:date_idx+1, symbol_idx]
                if np.any(np.diff(price_history) / price_history[:-1] > 0.15):
                    continue
                
                # Check if above short MA
                if close[date_idx, symbol_idx] < sma_short[date_idx, symbol_idx]:
                    continue
                
                # Check momentum
                if mom[date_idx, symbol_idx] < self._min_momentum:
                    continue
            
                qualified_stocks.append((symbol, mom[date_idx, symbol_idx]))
        
        return sorted(qualified_stocks, key=lambda x: x[1], reverse=True)
        
        
    def _rebalance_portfolio(self, ranked_stocks, mom, close, sma_short, date_idx):
        actual_date = next(date for date, idx in self.date_to_index.items() if idx == date_idx)
        # Sell all current positions not in top 20%
        top_20_percent = set([s[0] for s in ranked_stocks[:int(len(ranked_stocks)*0.2)]])
        for symbol in list(self.positions.keys()):
            if symbol not in top_20_percent and self.positions[symbol].get_current_quantity() > 0:
                self.sell(symbol, self.positions[symbol].get_current_quantity())
                
        for symbol in list(self.positions.keys()):
            if symbol in self.const_d2i[actual_date] and self.positions[symbol].get_current_quantity() > 0:
                symbol_idx = self.const_d2i[actual_date][symbol]
                
                price_history = close[date_idx-self.lookback:date_idx+1, symbol_idx]
                if np.any(np.diff(price_history) / price_history[:-1] > 0.15):
                    self.sell(symbol, self.positions[symbol].get_current_quantity())
                    continue
                
                # Check if above short MA
                if close[date_idx, symbol_idx] < sma_short[date_idx, symbol_idx]:
                    self.sell(symbol, self.positions[symbol].get_current_quantity())
                    continue
                
                # Check momentum
                if mom[date_idx, symbol_idx] < self._min_momentum:
                    self.sell(symbol, self.positions[symbol].get_current_quantity())
                    continue
         
    
    def _calculate_position_size(self, symbol: str, close: float, atr: float):
        risk_per_stock = self.broker.get_portfolio_value() * 0.002 # 2% of the portfolio at risk
        return int(risk_per_stock / atr)
    
    def _rebalance_positions(self, ranked_stocks: List[Tuple[str, float]], close: np.ndarray, atr_20: np.ndarray, sma_long: np.ndarray, sma_short: np.ndarray, date_idx: int):
        actual_date = next(date for date, idx in self.date_to_index.items() if idx == date_idx)
        
        # Adjust existing positions
        for symbol in list(self.positions.keys()):
            current_size = self.positions[symbol].get_current_quantity()
            if current_size <= 0 or symbol not in list(self.const_d2i[actual_date].keys()):
                continue
            symbol_idx = self.const_d2i[actual_date][symbol]
            target_size = self._calculate_position_size(symbol, close[date_idx, symbol_idx], atr_20[date_idx, symbol_idx])
            if abs(current_size - target_size) / current_size > 0.1:  # If difference is more than 10%
                if current_size < target_size:
                    self.buy(symbol, abs(target_size - current_size))
                elif current_size > target_size:
                    self.sell(symbol, abs(current_size - target_size))
        
        current_asset_count = 0
        for sym, pos in self.positions.items():
            if pos.get_current_quantity() > 0:
                current_asset_count += 1
        
        # Buy new positions if cash available and index above 200 EMA (assumes the last series is the index)
        if self.broker.get_disposable_cash() > 0 and self.index_historical[date_idx] > self.index_historical_mean[date_idx] and current_asset_count < self.max_stocks:
            for symbol, _ in ranked_stocks:
                if symbol not in self.positions or self.positions[symbol].get_current_quantity() <= 0:
                    symbol_idx = self.const_d2i[actual_date][symbol]
                    if close[date_idx, symbol_idx] > sma_short[date_idx, symbol_idx]:
                        size = self._calculate_position_size(symbol, close[date_idx, symbol_idx], atr_20[date_idx, symbol_idx])
                        if self.broker.get_disposable_cash() >= close[date_idx, symbol_idx] * size:
                            self.buy(symbol, size)
                        else:
                            break  # No more cash available
                        
    def _sell_non_members(self, date_idx: int):
        actual_date = next(date for date, idx in self.date_to_index.items() if idx == date_idx)
        for symbol in list(self.positions.keys()):
            current_size = self.positions[symbol].get_current_quantity()
            
            if current_size > 0 and symbol not in list(self.const_d2i[actual_date].keys()):
                self.sell(symbol, self.positions[symbol].get_current_quantity())

    
    def generate_signals(self, date_idx: int, indicator_calculator: IndicatorCalculator,
                         positions: Dict[str, Position], symbol_to_index: Dict[str, int], benchmark: pd.DataFrame):
        if date_idx < self.long_window:
            return  # Not enough data to generate signals

        if self.index_historical is None:
            self.index_historical = benchmark['Close'].to_numpy()
            self.index_historical_mean = benchmark['Close'].rolling(window=self.long_window).mean()

        close = indicator_calculator.get_indicator(f'Close')
        mom = indicator_calculator.get_indicator(f'Momentum_{self.lookback}')
        sma_short = indicator_calculator.get_indicator(f'SMA_{self.short_window}')
        atr_20 = indicator_calculator.get_indicator(f'ATR_{self.volatility}')
        sma_long = indicator_calculator.get_indicator(f'SMA_{self.long_window}')
              
        for symbol, symbol_idx in symbol_to_index.items():
            if symbol not in self.initialized:
                if symbol in positions:
                    self.initialized[symbol] = True
                    self.positions[symbol] = positions[symbol]
            
        self.signals = {}  # New signals to populate
        
        ranked_stocks = self._make_rankings(close, mom, sma_short, date_idx)
        
        self._sell_non_members(date_idx)
        
        if date_idx % 5 == 0:  # Approximating rebalancing
            self._rebalance_portfolio(ranked_stocks, mom, close, sma_short, date_idx)
            
        if date_idx % 10 == 0:
            self._rebalance_positions(ranked_stocks, close, atr_20, sma_long, sma_short, date_idx)
