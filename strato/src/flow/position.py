import datetime
import logging
from typing import List, Dict, Tuple
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
        self.open_lots = []
        self.current_quantity = 0
        if initial_quantity > 0:
            self.open_lots.append({
                'quantity': initial_quantity,
                'price': initial_price,
                'date': date_entered
            })
            self.current_quantity = initial_quantity

    def buy(self, quantity: int, price: float, date: datetime.datetime):
        """
        Buy more of the asset, increasing the quantity of the position.

        Args:
            quantity (int): The quantity to buy.
            price (float): The price per unit for this purchase.
            date (datetime.datetime): The date and time of this purchase.
        """
        self.open_lots.append({
            'quantity': quantity,
            'price': price,
            'date': date
        })
        self.current_quantity += quantity

    def sell(self, quantity: int, price: float, date: datetime.datetime) -> Tuple[float, List[Dict]]:
        """
        Sell some of the asset, decreasing the quantity of the position.
        This method uses a FIFO (First In, First Out) method when selling.

        Args:
            quantity (int): The quantity to sell.
            price (float): The price per unit for this sale.
            date (datetime.datetime): The date and time of this sale.

        Returns:
            Tuple[float, List[Dict]]: The realized profit and loss (PnL) from this sale,
                                      and a list of closed trade lots.

        Raises:
            ValueError: If attempting to sell more than the current quantity.
        """
        realized_pnl = 0
        remaining_to_sell = quantity
        closed_lots = []

        while remaining_to_sell > 0 and self.open_lots:
            lot = self.open_lots[0]
            if lot['quantity'] > remaining_to_sell:
                # Partial lot closure
                realized_pnl += remaining_to_sell * (price - lot['price'])
                closed_lots.append({
                    'buy_date': lot['date'],
                    'sell_date': date,
                    'quantity': remaining_to_sell,
                    'buy_price': lot['price'],
                    'sell_price': price,
                    'pnl': remaining_to_sell * (price - lot['price'])
                })
                lot['quantity'] -= remaining_to_sell
                self.current_quantity -= remaining_to_sell
                remaining_to_sell = 0
            else:
                # Full lot closure
                realized_pnl += lot['quantity'] * (price - lot['price'])
                closed_lots.append({
                    'buy_date': lot['date'],
                    'sell_date': date,
                    'quantity': lot['quantity'],
                    'buy_price': lot['price'],
                    'sell_price': price,
                    'pnl': lot['quantity'] * (price - lot['price'])
                })
                remaining_to_sell -= lot['quantity']
                self.current_quantity -= lot['quantity']
                self.open_lots.pop(0)

        if remaining_to_sell > 0:
            logging.error(f"Not enough quantity to sell for target quantity: {quantity}")
            raise ValueError(f"Not enough quantity to sell for {quantity} for {self.asset}. Check logs")

        return realized_pnl, closed_lots

    def calculate_value(self, current_price: float) -> float:
        """
        Calculate the current value of the position.

        Args:
            current_price (float): The current price per unit of the asset.

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
        return sum((current_price - lot['price']) * lot['quantity'] for lot in self.open_lots)

    def get_open_lots(self) -> List[Dict]:
        """
        Get a list of all open lots in the position.

        Returns:
            List[Dict]: A list of dictionaries, each representing an open lot with quantity, price, and date.
        """
        return self.open_lots

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
        return self.asset

    def __repr__(self):
        """
        Return a string representation of the Position.

        Returns:
            str: A string representation of the Position object.
        """
        return f"Position(symbol={self.asset}, quantity={self.current_quantity})"