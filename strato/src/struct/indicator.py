from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple

class Indicator(ABC):
    """
    Abstract base class for technical indicators.
    """

    @abstractmethod
    def init(self, data: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Initialize the indicator with historical data.

        Args:
            data (np.ndarray): Historical data to initialize the indicator.

        Returns:
            Tuple[np.ndarray, int]: 
                - Initial values of the indicator
                - Starting position for calculations
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, current_value: np.ndarray, new_data: np.ndarray, carry: np.ndarray) -> np.ndarray:
        """
        Update the indicator with new data.

        Args:
            current_value (np.ndarray): The current value of the data.
            new_data (np.ndarray): New data point(s) to update the indicator.
            carry (np.ndarray): The carry-over values from the previous step.

        Returns:
            np.ndarray: Updated value of the indicator.
        """
        raise NotImplementedError

class IndicatorCalculator:
    """
    A class to manage and calculate multiple technical indicators.
    """
    def __init__(self, data: np.ndarray, feature_to_index: Dict[str, int]):
        """
        Initialize the IndicatorCalculator.
        Args:
            data (np.ndarray): Historical market data.
            feature_to_index (Dict[str, int]): Mapping of feature names to their indices in the data.
        """
        self.data = data
        self.feature_to_index = feature_to_index
        self.indicators: Dict[str, Tuple[Indicator, int]] = {}
        self.indicator_values: Dict[str, np.ndarray] = {}
        self.carrys: Dict[str, np.ndarray] = {}

    def add_indicator(self, name: str, indicator: Indicator, column: str = 'Close'):
        """
        Add a new indicator to be calculated.
        Args:
            name (str): Name of the indicator.
            indicator (Indicator): Indicator object.
            column (str, optional): Data column to use for the indicator. Defaults to 'Close'.
        """
        column_index = self.feature_to_index[column]
        values = self.data[:, :, column_index]
        _, initial_value, start_position = indicator.init(values)
        self.carrys[name] = _
        self.indicators[name] = (indicator, start_position)
        self.indicator_values[name] = np.full((self.data.shape[0], self.data.shape[1]), np.nan)
        self.indicator_values[name][:start_position] = initial_value

    def calculate_indicators(self) -> int:
        """
        Calculate all added indicators for the entire dataset.
        Returns:
            int: The maximum start position among all indicators.
        """
        start_positions = []
        column_index = self.feature_to_index['Close']  # Assuming 'Close' for now
        values = self.data[:, :, column_index]
        for name, (indicator, start_position) in self.indicators.items():
            carry = self.carrys[name]  # Initialize carry with the last pre-calculated value
            for i in range(start_position, self.data.shape[0]):
                current_value = values[i - 1]
                new_data = values[i]
                carry, updated_value = indicator.step(current_value, new_data, carry)
                self.indicator_values[name][i] = updated_value
            start_positions.append(start_position)
        return max(start_positions)

    def get_indicator(self, name: str) -> np.ndarray:
        """
        Get the calculated values for a specific indicator.
        Args:
            name (str): Name of the indicator.
        Returns:
            np.ndarray: Calculated values of the indicator.
        """
        return self.indicator_values.get(name)

