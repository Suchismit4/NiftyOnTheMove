import numpy as np
from typing import Dict, Tuple
import sys
from strato.src.struct.indicator import Indicator

class Close(Indicator):
    def __init__(self, log: bool = False):
        """
        Initialize the Close price indicator.
        
        Args:
            log (bool): Whether to apply logarithmic transformation to prices. Default is False.
        """
        self.log = log

    def _apply_log(self, x: np.ndarray) -> np.ndarray:
        """
        Apply logarithmic transformation if self.log is True.
        
        Args:
            x (np.ndarray): Input price data.
        
        Returns:
            np.ndarray: Log-transformed data if self.log is True, otherwise original data.
        """
        if self.log:
            return np.log(x)
        return x

    def init(self, data: np.ndarray, feature_to_index: Dict) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Initialize the Close indicator using the provided historical data.
        
        Args:
            data (np.ndarray): Historical price data with shape (time_steps, num_assets).
        
        Returns:
            Tuple[np.ndarray, np.ndarray, int]: A tuple containing:
                - carry (np.ndarray): Empty array as no carry is needed.
                - prices (np.ndarray): The last 'window' close prices for each asset.
                - warmup_period (int): The number of periods used for initialization.
        """
        
        self.feature_to_index = feature_to_index
        # data shape: (time_steps, num_assets)
        data = data[:, :, self.feature_to_index['Close']]
        prices = self._apply_log(data[0])
        carry = np.array([])  # No carry needed for this indicator
        return carry, prices, 1

    def step(self, current_value: np.ndarray, new_data: np.ndarray, carry: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update the Close indicator with new data.
        
        Args:
            current_value (np.ndarray): The current price for each asset. Shape: (num_assets,)
            new_data (np.ndarray): The new price data for each asset. Shape: (num_assets,)
            carry (np.ndarray): Empty array as no carry is needed.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - carry (np.ndarray): Empty array as no carry is needed.
                - new_prices (np.ndarray): The updated close prices. Shape: (window, num_assets)
        """
        new_data = new_data[:, self.feature_to_index['Close']]
        new_price = self._apply_log(new_data)
        
        return carry, new_price
    
    
class ExponentialMovingAverage(Indicator):
    def __init__(self, window: int):
        """
        Initialize the Exponential Moving Average indicator.
        
        Args:
            window (int): The number of periods to consider for the EMA calculation.
        """
        self.window = window
        self.alpha = 2 / (window + 1)  # Smoothing factor for EMA

    def init(self, data: np.ndarray, feature_to_index: Dict) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Initialize the EMA calculation using the provided historical data.
        
        Args:
            data (np.ndarray): Historical price data with shape (time_steps, num_assets).
        
        Returns:
            Tuple[np.ndarray, np.ndarray, int]: A tuple containing:
                - carry (np.ndarray): The last EMA value for each asset.
                - ema (np.ndarray): The initial EMA value for each asset.
                - warmup_period (int): The number of periods used for initialization.
        """
        # data shape: (time_steps, num_assets)
        if data.shape[0] < self.window:
            return np.full((1, data.shape[1]), np.nan), np.full(data.shape[1], np.nan), 0
        self.feature_to_index = feature_to_index
        data = data[:, :, self.feature_to_index['Close']]

        # Calculate the initial SMA as the first EMA value
        # initial_sma shape: (num_assets,)
        initial_sma = np.mean(data[:self.window], axis=0)

        # carry shape: (1, num_assets)
        carry = initial_sma.reshape(1, -1)

        return carry, initial_sma, self.window

    def step(self, current_value: np.ndarray, new_data: np.ndarray, carry: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update the EMA with new data.
        
        Args:
            current_value (np.ndarray): The current price for each asset. Shape: (num_assets,)
            new_data (np.ndarray): The new price data for each asset. Shape: (num_assets,)
            carry (np.ndarray): The previous EMA value for each asset. Shape: (1, num_assets)
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - new_carry (np.ndarray): The updated EMA value for each asset. Shape: (1, num_assets)
                - new_ema (np.ndarray): The new EMA value for each asset. Shape: (num_assets,)
        """
        # Calculate new EMA
        # new_ema shape: (num_assets,)
        new_data = new_data[:, self.feature_to_index['Close']]
        current_value = current_value[:, self.feature_to_index['Close']]
        
        new_ema = self.alpha * new_data + (1 - self.alpha) * carry[0]

        # Update carry
        # new_carry shape: (1, num_assets)
        new_carry = new_ema.reshape(1, -1)

        return new_carry, new_ema

    
class AverageTrueRange(Indicator):
    def __init__(self, window: int):
        """
        Initialize the Average True Range (ATR) indicator.
        
        Args:
            window (int): The number of periods to consider for the ATR calculation.
        """
        self.window = window

    def init(self, data: np.ndarray, feature_to_index: Dict) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Initialize the ATR calculation using the provided historical data.
        
        Args:
            data (np.ndarray): Historical price data with shape (time_steps, num_assets, 3) where the last dimension 
                               contains (high, low, close) prices.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, int]: A tuple containing:
                - carry (np.ndarray): The last ATR value for each asset.
                - atr (np.ndarray): The initial ATR value for each asset.
                - warmup_period (int): The number of periods used for initialization.
        """
        if data.shape[0] < self.window + 1:
            return np.full((1, data.shape[1]), np.nan), np.full(data.shape[1], np.nan), 0
        self.feature_to_index = feature_to_index
        high = data[:self.window, :, self.feature_to_index['High']]
        low = data[:self.window, :, self.feature_to_index['Low']]
        close = data[:self.window, :, self.feature_to_index['Close']]

        true_range = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        atr_initial = np.mean(true_range, axis=0)

        atr = np.zeros_like(atr_initial)
        for i in range(-self.window + 1, 0):
            atr = (atr * (self.window - 1) + true_range[i]) / self.window

        carry = atr.reshape(1, -1)

        return carry, atr, self.window

    def step(self, current_value: np.ndarray, new_data: np.ndarray, carry: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update the ATR with new data.
        
        Args:
            current_value (np.ndarray): The current price data for each asset. Shape: (num_assets, 3)
            new_data (np.ndarray): The new price data for each asset. Shape: (num_assets, 3)
            carry (np.ndarray): The previous ATR value for each asset. Shape: (1, num_assets)
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - new_carry (np.ndarray): The updated ATR value for each asset. Shape: (1, num_assets)
                - new_atr (np.ndarray): The new ATR value for each asset. Shape: (num_assets,)
        """
        high = new_data[:, self.feature_to_index['High']]
        low = new_data[:, self.feature_to_index['Low']]
        close_prev = current_value[:, self.feature_to_index['Close']]

        true_range = np.maximum(high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev)))
        new_atr = (carry[0] * (self.window - 1) + true_range) / self.window

        new_carry = new_atr.reshape(1, -1)

        return new_carry, new_atr
    
    
class Volatility(Indicator):
    def __init__(self, window: int = 20, annualize: bool = True, ewma: bool = False, parkinson: bool = False):
        """
        Initialize the Volatility indicator.
        
        Args:
            window (int): The number of periods to consider for volatility calculation. Default is 20.
            annualize (bool): Whether to annualize the volatility. Default is True.
            ewma (bool): Whether to use Exponentially Weighted Moving Average. Default is False.
            parkinson (bool): Whether to use Parkinson's volatility estimator. Default is False.
        """
        self.window = window
        self.annualize = annualize
        self.ewma = ewma
        self.parkinson = parkinson
        
        if self.ewma:
            self.lambda_param = 0.94  # EWMA decay factor, typically 0.94 for daily data
        
        self.sqrt_252 = np.sqrt(252)  # Annualization factor for daily data

    def init(self, data: np.ndarray, feature_to_index: Dict) -> Tuple[np.ndarray, np.ndarray, int]:
        self.feature_to_index = feature_to_index
        
        if data.shape[0] < self.window:
            return np.full((self.window, data.shape[1]), np.nan), np.full(data.shape[1], np.nan), 0
        
        close_prices = data[:self.window, :, self.feature_to_index['Close']]
        
        if self.parkinson:
            high_prices = data[:self.window, :, self.feature_to_index['High']]
            low_prices = data[:self.window, :, self.feature_to_index['Low']]
            log_hl = np.log(high_prices / low_prices)
            vol = np.sqrt(np.sum(log_hl**2, axis=0) / (4 * self.window * np.log(2)))
        else:
            returns = np.log(close_prices[1:] / close_prices[:-1])
            
            if self.ewma:
                weights = (1 - self.lambda_param) * self.lambda_param**np.arange(self.window-1)[::-1]
                weights /= np.sum(weights)
                vol = np.sqrt(np.sum(weights * returns**2, axis=0))
            else:
                vol = np.std(returns, axis=0)
        
        if self.annualize:
            vol *= self.sqrt_252
        
        carry = np.vstack([close_prices])
        
        return carry, vol, self.window

    def step(self, current_value: np.ndarray, new_data: np.ndarray, carry: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        new_close = new_data[:, self.feature_to_index['Close']]
        close_prices = np.vstack((carry[1:], new_close))
        
        if self.parkinson:
            new_high = new_data[:, self.feature_to_index['High']]
            new_low = new_data[:, self.feature_to_index['Low']]
            log_hl = np.log(new_high / new_low)
            vol = np.sqrt(log_hl**2 / (4 * np.log(2)))
        else:
            returns = np.log(close_prices[1:] / close_prices[:-1])
            
            if self.ewma:
                weights = (1 - self.lambda_param) * self.lambda_param**np.arange(self.window-1)[::-1]
                weights /= np.sum(weights)
                vol = np.sqrt(np.sum(weights * returns**2, axis=0))
            else:
                vol = np.std(returns, axis=0)
        
        if self.annualize:
            vol *= self.sqrt_252
        
        carry = np.vstack([close_prices])
        
        return carry, vol
    
    
class Momentum(Indicator):
    def __init__(self, window: int = 90, inverse_logistic: bool = False):
        self.window = window
        self.inverse_logistic = inverse_logistic

    def _apply_inverse_logistic(self, x: np.ndarray) -> np.ndarray:
        if self.inverse_logistic:
            return -10. * np.log(2. / (1 + 0.00999 * (2 * x - 100)) - 1)
        return x

    def init(self, data: np.ndarray, feature_to_index: Dict) -> Tuple[np.ndarray, np.ndarray, int]:
        self.feature_to_index = feature_to_index
        
        self.x = np.arange(self.window).reshape(-1, 1)
        self.x = np.tile(self.x, (1, data.shape[1]))
        
        self.xmean = np.mean(self.x, axis=0)
        self.x_dev = self.x - self.xmean
        
        y = data[:self.window, :, feature_to_index['Close']]
        y = np.log(y)
        
        if y.shape[0] < self.window:
            return np.full((self.window, data.shape[1]), np.nan), np.full(data.shape[1], np.nan), 0
        
        ymean = np.mean(y, axis=0)
        y_dev = y - ymean
        
        # Compute covariance components
        ssxm = np.sum(self.x_dev ** 2, axis=0)
        ssxym = np.sum(self.x_dev * y_dev, axis=0)
        ssym = np.sum(y_dev ** 2, axis=0)
        
        if np.any(ssxm) == 0.0 or np.any(ssym) == 0.0:
            # If the denominator was going to be 0
            r = 0.0
        else:
            r = ssxym / np.sqrt(ssxm * ssym)

        slope = ssxym / ssxm
        
        slope = ssxym / ssxm
        annualized = ((np.exp(slope * 252)) - 1) * 100
        momentum = annualized * (r ** 2)

        trend = self._apply_inverse_logistic(momentum)
        carry = np.vstack([y])  
        
        return carry, trend, self.window

    def step(self, current_value: np.ndarray, new_data: np.ndarray, carry: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        new_close = new_data[:, self.feature_to_index['Close']]
        new_log_price = np.log(new_close)
        # Update the carry with new data point new_log_price
        y = np.vstack((carry[1:], new_log_price))
        
        ymean = np.mean(y, axis=0)
        y_dev = y - ymean
        
        # Compute covariance components
        ssxm = np.sum(self.x_dev ** 2, axis=0)
        ssxym = np.sum(self.x_dev * y_dev, axis=0)
        ssym = np.sum(y_dev ** 2, axis=0)
        
        if np.any(ssxm) == 0.0 or np.any(ssym) == 0.0:
            # If the denominator was going to be 0
            r = 0.0
        else:
            r = ssxym / np.sqrt(ssxm * ssym)
            # Test for numerical error propagation (make sure -1 < r < 1)

        slope = ssxym / ssxm
        
        slope = ssxym / ssxm
        annualized = ((np.exp(slope * 252)) - 1) * 100
        momentum = annualized * (r ** 2)

        trend = self._apply_inverse_logistic(momentum)
        carry = np.vstack([y])  
        
        return carry, trend