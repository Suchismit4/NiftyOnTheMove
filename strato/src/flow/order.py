import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime
from typing import Tuple
from ..struct.strategy import Strategy

class Order:
    """Represents a trading order."""

    def __init__(self, order_type: int, order_date_idx: int, quantity: int, position: object, symbol: int):
        """
        Initialize an Order object.

        Args:
            order_type (int): Type of order (BUY or SELL).
            order_date_idx (int): Index of the order date.
            quantity (int): Quantity of the asset to trade.
            position (object): Position object associated with this order.
            symbol (int): Symbol index for the asset.
        """
        self.order_type = order_type
        self.order_date_idx = order_date_idx
        self.quantity = quantity
        self.symbol = symbol
        self.position = position

    def execute(self, daily_data: np.ndarray, feature_to_index: dict, order_date) -> Tuple[float, int, float, int]:
        """
        Execute the order.

        Args:
            daily_data (np.ndarray): Daily market data.
            feature_to_index (dict): Mapping of features to their indices.
            order_date: Date of the order execution.

        Returns:
            Tuple[float, int, float, int]: 
                - Value of the trade
                - Quantity traded
                - Execution price
                - Order type
        """
        execution_price = daily_data[self.symbol, feature_to_index['Open']]
        
        if self.order_type == Strategy.BUY:
            value = -1 * self.quantity * execution_price
            self.position.buy(self.quantity, execution_price, order_date)
        elif self.order_type == Strategy.SELL:
            value = self.quantity * execution_price
            self.position.sell(self.quantity, execution_price, order_date)
        
        return value, self.quantity, execution_price, self.order_type