import datetime
import logging
from typing import List, Dict
import numpy as np

class Position:
    """
    Represents a trading position for a specific asset.
    
    This class manages the quantity, price, and date information for a trading position,
    allowing for adjustments and calculations related to the position.
    """

    def __init__(self, asset: str, initial_quantity: int, initial_price: float, date_entered: datetime.datetime):
        """
        Initialize a new Position.

        Args:
            asset (str): The identifier for the asset (e.g., stock symbol).
            initial_quantity (int): The initial quantity of the asset.
            initial_price (float): The initial price per unit of the asset.
            date_entered (datetime.datetime): The date and time when the position was entered.
        """
        self.asset = asset
        self.bars_since_entry = 0
        self.entries = [{'quantity': initial_quantity, 'price': initial_price, 'date': date_entered}]
        self.current_quantity = initial_quantity
        logging.debug(f"Position initialized with entries: {self.entries} and current quantity: {self.current_quantity}")

    def buy(self, quantity: int, price: float, date: datetime.datetime):
        """
        Buy more of the asset, increasing the quantity of the position.

        Args:
            quantity (int): The quantity to buy.
            price (float): The price per unit for this purchase.
            date (datetime.datetime): The date and time of this purchase.
        """
        logging.debug(f"Buying more of asset: {self.asset}, quantity: {quantity}, price: {price}, date: {date}")
        self.entries.append({'quantity': quantity, 'price': price, 'date': date})
        self.current_quantity += quantity
        logging.debug(f"Bought more of {self.asset}: {quantity} units at {price} on {date}")

    def sell(self, quantity: int, price: float, date: datetime.datetime) -> float:
        """
        Sell some of the asset, decreasing the quantity of the position.
        This method uses a FIFO (First In, First Out) method when selling.

        Args:
            quantity (int): The quantity to sell.
            price (float): The price per unit for this sale.
            date (datetime.datetime): The date and time of this sale.

        Returns:
            float: The realized profit and loss (PnL) from this sale.

        Raises:
            ValueError: If attempting to sell more than the current quantity.
        """
        logging.debug(f"Selling asset: {self.asset}, quantity: {quantity}, price: {price}, date: {date}")
        realized_pnl = 0
        remaining_to_sell = quantity

        while remaining_to_sell > 0 and self.entries:
            entry = self.entries[0]
            if entry['quantity'] > remaining_to_sell:
                entry['quantity'] -= remaining_to_sell
                self.current_quantity -= remaining_to_sell
                realized_pnl += remaining_to_sell * (price - entry['price'])
                remaining_to_sell = 0
            else:
                realized_pnl += entry['quantity'] * (price - entry['price'])
                remaining_to_sell -= entry['quantity']
                self.current_quantity -= entry['quantity']
                self.entries.pop(0)

        logging.debug(f"Sold {quantity - remaining_to_sell} units, remaining to sell: {remaining_to_sell}")

        if remaining_to_sell > 0:
            logging.error(f"Not enough quantity to sell for target quantity: {quantity}")
            raise ValueError(f"Not enough quantity to sell for {quantity}. See logs")

        logging.debug(f"Quantity adjusted. Current quantity: {self.current_quantity}, Realized PnL: {realized_pnl}")
        return realized_pnl

    def calculate_value(self, current_price: np.ndarray) -> float:
        """
        Calculate the current value of the position.

        Args:
            current_price (np.ndarray): The current price per unit of the asset.

        Returns:
            float: The total value of the position.
        """
        return self.current_quantity * current_price

    def calculate_pnl(self, current_price: float) -> float:
        """
        Calculate the unrealized profit and loss (PnL) of the position.

        Args:
            current_price (float): The current price per unit of the asset.

        Returns:
            float: The unrealized PnL of the position.
        """
        return sum((current_price - entry['price']) * entry['quantity'] for entry in self.entries)

    def get_positions(self) -> List[Dict]:
        """
        Get a list of all entries in the position.

        Returns:
            List[Dict]: A list of dictionaries, each representing an entry with quantity, price, and date.
        """
        return self.entries

    def get_current_quantity(self) -> int:
        """
        Get the current total quantity of the position.

        Returns:
            int: The current quantity.
        """
        return self.current_quantity

    def get_asset(self) -> str:
        """
        Get the asset identifier for this position.

        Returns:
            str: The asset identifier (e.g., stock symbol).
        """
        logging.debug(f"Getting asset for position: {self.asset}")
        return self.asset

    def __repr__(self):
        """
        Return a string representation of the Position.

        Returns:
            str: A string representation of the Position object.
        """
        return f"Position(symbol={self.asset}, quantity={self.current_quantity})"