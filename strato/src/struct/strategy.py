import numpy as np
from abc import ABC, abstractmethod
from typing import Dict
import equinox as eqx
from ..flow.position import Position
import pandas as pd

class Broker():
    
    def __init__(self):
        self.cash = -np.inf
        self.portfolio_value = -np.inf
        pass

    def set(self, cash, portfolio_value): 
        self.cash = cash
        self.portfolio_value = portfolio_value
        
    def get_disposable_cash(self):
        return self.cash
    
    def get_portfolio_value(self):
        return self.portfolio_value

class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    """

    BUY = 1
    SELL = -1
    HOLD = 0

    def __init__(self):
        """
        Initialize the Strategy object.
        """
        self.signals = {}
        self.name = "A Strategy"
        self.broker = Broker()
    
    def __getname__(self):
        return self.name

    def buy(self, symbol: str, quantity = 10):
        """
        Set a buy signal for a given symbol.

        Args:
            symbol (str): The asset symbol.
        """
        self.signals[symbol] = [self.BUY, quantity]

    def sell(self, symbol: str, quantity = 10):
        """
        Set a sell signal for a given symbol.

        Args:
            symbol (str): The asset symbol.
        """
        self.signals[symbol] = [self.SELL, quantity]

    def hold(self, symbol: str):
        """
        Set a hold signal for a given symbol.

        Args:
            symbol (str): The asset symbol.
        """
        self.signals[symbol] = [self.HOLD, 0]

    def get_signals(self) -> Dict[str, Dict[int, int]]:
        """
        Get the current signals for all symbols.

        Returns:
            Dict[str, Dict[int, int]]: A dictionary of symbols and their corresponding signals.
        """
        return self.signals

    @abstractmethod
    def generate_signals(self, date_idx: int, indicator_calculator, positions: Dict[str, Position], symbol_to_index: Dict[str, int], benchmark: pd.DataFrame = None):
        """
        Generate trading signals based on the current market state.

        This method should be implemented by concrete strategy classes.

        Args:
            date_idx (int): The index of the current date.
            indicator_calculator: An object that calculates technical indicators.
            positions (Dict[str, Position]): A dictionary of current positions.
            symbol_to_index (Dict[str, int]): A mapping of symbols to their indices.
        """
        pass