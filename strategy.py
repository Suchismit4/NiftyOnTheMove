import sys
from strato.src.struct.strategy import Strategy
from strato.src.struct.indicator import IndicatorCalculator
from strato.src.flow.position import Position
from typing import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

class StocksOnTheMoveByAndrewsClenow(Strategy):
    def __init__(self,
                 constituents: pd.DataFrame,
                 symbol_to_index: Dict[str, int],
                 date_to_index: Dict[datetime.datetime, int],
                 lookback: int = 90,
                 sma_short: int = 100,
                 sma_long: int = 200,
                 volatility: int = 20,
                 min_cash: float = 100,
                 min_momentum: float = 0.,
                 trailing_stop_percent: float = 0.1,
                 max_stocks: int = 15,
                 portfolio_at_risk: float = 0.001):
        """
        Initializes the Clenow Momentum Strategy.

        Args:
            constituents (pd.DataFrame): The DataFrame containing the constituent information.
            symbol_to_index (Dict[str, int]): A dictionary mapping symbols to their indices.
            date_to_index (Dict[datetime.datetime, int]): A dictionary mapping dates to their indices.
            lookback (int): The lookback period for the momentum calculation.
            sma_short (int): The short-term Simple Moving Average window.
            sma_long (int): The long-term Simple Moving Average window.
            volatility (int): The window for the Average True Range (ATR) calculation.
            min_cash (float): The minimum cash required to open a new position.
            min_momentum (float): The minimum momentum required to open a new position.
            max_stocks (int): The maximum number of stocks to hold in the portfolio.
        """
        super().__init__()
        self.name = "Clenow Momentum Strategy (Long only)"
        self.short_window = sma_short
        self.long_window = sma_long
        self.min_momentum = min_momentum
        self.volatility = volatility
        self.lookback = lookback
        self.max_stocks = max_stocks
        self.constituents = constituents
        self.portfolio_at_risk = portfolio_at_risk
        self.index_historical = None
        self.index_historical_mean = None
        self.min_cash = min_cash
        self.constituents['Event Date'] = pd.to_datetime(self.constituents['Event Date'])

        # Create a dictionary mapping tickers to their indices
        self.ticker_to_index = {symbol: symbol_to_index.get(symbol, -1) for symbol in self.constituents['ticker'].unique()}

        # Group by 'Event Date' and aggregate the tickers
        grouped_tickers_by_date = self.constituents.groupby('Event Date')['ticker'].apply(list)

        # Create the final dictionary mapping dates to symbols and their indices
        self.date_to_symbols_with_indices = {
            date: {symbol: self.ticker_to_index[symbol] for symbol in tickers if self.ticker_to_index[symbol] != -1}
            for date, tickers in grouped_tickers_by_date.items()
        }

        self.date_to_index = date_to_index
        self.initialized: Dict[str, bool] = {}
        self.positions: Dict[str, Position] = {}
        
        self.trailing_stop_percent = trailing_stop_percent
        self.trailing_stops = {}

    def _make_rankings(self, close_prices: np.ndarray, momentum: np.ndarray, sma_short: np.ndarray, date_idx: int) -> List[Tuple[str, np.ndarray]]:
        """
        Generates a ranking of qualified stocks based on momentum.

        Args:
            close_prices (np.ndarray): The close prices for all stocks.
            momentum (np.ndarray): The momentum values for all stocks.
            sma_short (np.ndarray): The short-term SMA values for all stocks.
            date_idx (int): The index of the current date.

        Returns:
            List[Tuple[str, np.ndarray]]: A list of tuples containing the symbol and its momentum value.
        """
        qualified_stocks = []
        actual_date = self._get_actual_date(date_idx) 

        # Evaluate each stock to determine if it meets the criteria for ranking
        for symbol, position in self.positions.items():
            if symbol in self.date_to_symbols_with_indices[actual_date]:
                symbol_idx = self.date_to_symbols_with_indices[actual_date][symbol]
                price_history = close_prices[date_idx - self.lookback:date_idx + 1, symbol_idx]

                # Exclude stocks with more than 15% price change
                if np.any(np.diff(price_history) / price_history[:-1] > 0.15):
                    continue
                
                # Exclude stocks below the short-term SMA
                if close_prices[date_idx, symbol_idx] < sma_short[date_idx, symbol_idx]:
                    continue
                
                # Exclude stocks below the minimum momentum
                if momentum[date_idx, symbol_idx] < self.min_momentum:
                    continue
                
                # Add qualified stocks to the list
                qualified_stocks.append((symbol, momentum[date_idx, symbol_idx]))

        # Sort the stocks by momentum in descending order
        return sorted(qualified_stocks, key=lambda x: x[1], reverse=True)

    def _update_trailing_stop(self, symbol: str, close_prices: np.ndarray, date_idx: int):
        """
        Updates the trailing stop level for a given symbol.

        Args:
            symbol (str): The symbol of the stock.
            close_prices (np.ndarray): The close prices for all stocks.
            date_idx (int): The index of the current date.
        """
        raise NotImplementedError

    def _rebalance_portfolio(self, ranked_stocks: List[Tuple[str, float]], momentum: np.ndarray, close_prices: np.ndarray, sma_short: np.ndarray, date_idx: int, atr_20):
        """
        Rebalances the portfolio by selling positions that are not in the top 20% or no longer meet the criteria.

        Args:
            ranked_stocks (List[Tuple[str, float]]): The list of qualified stocks ranked by momentum.
            momentum (np.ndarray): The momentum values for all stocks.
            close_prices (np.ndarray): The close prices for all stocks.
            sma_short (np.ndarray): The short-term SMA values for all stocks.
            date_idx (int): The index of the current date.
        """
        actual_date = self._get_actual_date(date_idx)
        top_20_percent_stocks = set([symbol for symbol, _ in ranked_stocks[:int(len(ranked_stocks) * 0.20)]])

        # Evaluate each stock in the current positions
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            symbol_idx = self.date_to_symbols_with_indices[actual_date].get(symbol)

            # Sell positions not in the top 20%
            if symbol not in top_20_percent_stocks and position.get_current_quantity() > 0:
                self.sell(symbol, position.get_current_quantity())
            elif symbol_idx is not None and position.get_current_quantity() > 0:
                price_history = close_prices[date_idx - self.lookback:date_idx + 1, symbol_idx]

                # Sell positions with more than 50% price change
                if np.any(np.diff(price_history) / price_history[:-1] > 0.5):
                    self.sell(symbol, position.get_current_quantity())
                
                # Sell positions below the short-term SMA
                elif close_prices[date_idx, symbol_idx] < sma_short[date_idx, symbol_idx]:
                    self.sell(symbol, position.get_current_quantity())
                
                # Sell positions below the minimum momentum
                elif momentum[date_idx, symbol_idx] < self.min_momentum:
                    self.sell(symbol, position.get_current_quantity())
                

        current_asset_count = sum(1 for pos in self.positions.values() if pos.get_current_quantity() > 0)

        # Buy new positions if cash is available and the number of positions is less than the maximum
        if self.broker.get_disposable_cash() > 0 and current_asset_count < self.max_stocks:
            current_disposable_cash = self.broker.get_disposable_cash()
            for symbol, factor in ranked_stocks:
                if self.positions[symbol].get_current_quantity() <= 0:
                    symbol_idx = self.date_to_symbols_with_indices[actual_date][symbol]
                    if close_prices[date_idx, symbol_idx] > sma_short[date_idx, symbol_idx]:
                        size = self._calculate_position_size(symbol, close_prices[date_idx, symbol_idx], atr_20[date_idx, symbol_idx])
                        if current_disposable_cash >= close_prices[date_idx, symbol_idx] * size and self.index_historical[date_idx] > self.index_historical_mean[date_idx]:
                            self.buy(symbol, size)
                            current_asset_count += 1
                            current_disposable_cash -= close_prices[date_idx, symbol_idx] * size
                        else:
                            break  # No more cash available

    def _calculate_position_size(self, symbol: str, close_price: float, atr: float) -> int:
        """
        Calculates the target position size for a given symbol.

        Args:
            symbol (str): The symbol of the stock.
            close_price (float): The current close price of the stock.
            atr (float): The current ATR value of the stock.

        Returns:
            int: The target position size.
        """
        risk_per_stock = self.broker.get_portfolio_value() * self.portfolio_at_risk  # Example: 1.5% of the portfolio at risk
        return int(risk_per_stock / atr)

    def _rebalance_positions(self, ranked_stocks: List[Tuple[str, float]], close_prices: np.ndarray, atr_20: np.ndarray, momentum: np.ndarray, sma_short: np.ndarray, date_idx: int):
        """
        Rebalances the positions in the portfolio based on the ranked stocks and other indicators.

        Args:
            ranked_stocks (List[Tuple[str, float]]): The list of qualified stocks ranked by momentum.
            close_prices (np.ndarray): The close prices for all stocks.
            atr_20 (np.ndarray): The ATR values for all stocks.
            sma_long (np.ndarray): The long-term SMA values for all stocks.
            sma_short (np.ndarray): The short-term SMA values for all stocks.
            date_idx (int): The index of the current date.
        """
        actual_date = self._get_actual_date(date_idx)
            
        # Adjust existing positions
        for symbol, position in self.positions.items():
            current_size = position.get_current_quantity()
            symbol_idx = self.date_to_symbols_with_indices[actual_date].get(symbol)

            if current_size <= 0 or symbol_idx is None:
                continue

            target_size = self._calculate_position_size(symbol, close_prices[date_idx, symbol_idx], atr_20[date_idx, symbol_idx])

            # If the size difference is more than 5%, adjust the position
            if abs(current_size - target_size) / current_size > 0.05:
                if current_size < target_size and close_prices[date_idx, symbol_idx] > sma_short[date_idx, symbol_idx]:
                    self.buy(symbol, abs(target_size - current_size)) 
                elif current_size > target_size:
                    self.sell(symbol, abs(current_size - target_size))

    def _sell_non_members(self, date_idx: int):
        """
        Sells positions in stocks that are no longer in the constituents list.

        Args:
            date_idx (int): The index of the current date.
        """
        actual_date = self._get_actual_date(date_idx)
        for symbol, position in list(self.positions.items()):
            if position.get_current_quantity() > 0 and symbol not in self.date_to_symbols_with_indices[actual_date]:
                self.sell(symbol, position.get_current_quantity())

    def step(self, date_idx: int, indicator_calculator, positions: Dict[str, Position], symbol_to_index: Dict[str, int], benchmark: pd.DataFrame = None):
        """
        Generates the trading signals for the current date.

        Args:
            date_idx (int): The index of the current date.
            indicator_calculator (IndicatorCalculator): The object responsible for calculating the necessary indicators.
            positions (Dict[str, Position]): The current positions in the portfolio.
            symbol_to_index (Dict[str, int]): A dictionary mapping symbols to their indices.
            benchmark (pd.DataFrame): The benchmark DataFrame containing the index historical data.
        """
        if date_idx < self.long_window:
            return  # Not enough data to generate signals

        # Initialize historical index data if not already done
        if self.index_historical is None:
            self.index_historical = benchmark['Close'].to_numpy()
            self.index_historical_mean = benchmark['Close'].rolling(window=self.long_window).mean()

        # Retrieve necessary indicators
        close_prices = indicator_calculator.get_indicator('Close')
        momentum = indicator_calculator.get_indicator(f'Momentum_{self.lookback}')
        sma_short = indicator_calculator.get_indicator(f'SMA_{self.short_window}')
        atr_20 = indicator_calculator.get_indicator(f'ATR_{self.volatility}')
        sma_long = indicator_calculator.get_indicator(f'SMA_{self.long_window}')

        # Initialize positions if not already done
        for symbol, symbol_idx in symbol_to_index.items():
            if symbol not in self.initialized:
                if symbol in positions:
                    self.initialized[symbol] = True
                    self.positions[symbol] = positions[symbol]

        self.signals = {}

        # Generate rankings for the stocks based on momentum
        ranked_stocks = self._make_rankings(close_prices, momentum, sma_short, date_idx)
        ranked_stocks = [(symbol, factor) for symbol, factor in ranked_stocks if not np.isnan(factor)]

        # Sell positions in stocks that are no longer in the constituents list
        self._sell_non_members(date_idx)

        # Rebalance the positions every 5
        if date_idx % 5 == 0:
            self._rebalance_positions(ranked_stocks, close_prices, atr_20, momentum, sma_short, date_idx)

        # Rebalance the portfolio every 10
        if date_idx % 10 == 0:
            self._rebalance_portfolio(ranked_stocks, momentum, close_prices, sma_short, date_idx, atr_20)

        
    def _get_actual_date(self, date_idx: int) -> pd.Timestamp:
        """
        Helper function to get the actual date corresponding to a date index.

        Args:
            date_idx (int): The index of the current date.

        Returns:
            pd.Timestamp: The actual date corresponding to the index.
        """
        return next(date for date, idx in self.date_to_index.items() if idx == date_idx)
