# Standard library imports
import datetime
import logging
import os
from typing import Dict, List
import tempfile
import subprocess
import shutil

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# Local imports
from .struct.strategy import Strategy
from .flow.position import Position
from .struct.indicator import IndicatorCalculator, Indicator
from .flow.order import Order

class Strato:
    """
    Main class for backtesting trading strategies.
    
    This class handles the simulation of trading, including position management,
    order execution, and performance tracking.
    """
    
    def __init__(self, data: np.ndarray, symbol_to_index: Dict[str, int], 
                 feature_to_index: Dict[str, int], date_to_index: Dict[datetime.datetime, int], 
                 starting_cash: float, trade_size: int, strategy: Strategy):
        """
        Initialize the Strato backtesting environment.

        Args:
            data (np.ndarray): Historical market data.
            symbol_to_index (Dict[str, int]): Mapping of symbols to their indices in the data array.
            feature_to_index (Dict[str, int]): Mapping of features to their indices in the data array.
            date_to_index (Dict[datetime.datetime, int]): Mapping of dates to their indices in the data array.
            starting_cash (float): Initial cash balance.
            trade_size (int): Number of units per trade.
            strategy (Strategy): Trading strategy to be tested.
        """
        logging.debug("Initializing Strato class")
        
        self.data = data
        self.symbol_to_index = symbol_to_index
        self.feature_to_index = feature_to_index
        self.date_to_index = date_to_index
        self.starting_cash = starting_cash
        self.trade_size = trade_size
        self.strategy = strategy
        
        # Initialize positions, cash, and orders
        self.positions = {symbol: Position(symbol, 0, 0, datetime.datetime.now()) for symbol in symbol_to_index.keys()}
        self.cash = starting_cash
        self.orders = {symbol: [] for symbol in symbol_to_index.keys()}
        
        # Initialize indicator calculator and last valid prices
        self.indicator_calculator = IndicatorCalculator(data, feature_to_index)
        self.last_valid_price = {symbol: None for symbol in symbol_to_index.keys()}
        
        logging.debug(f"Initialized with data shape: {data.shape}, starting cash: {starting_cash}, trade size: {trade_size}")
        logging.debug(f"Symbols: {list(symbol_to_index.keys())}, Features: {list(feature_to_index.keys())}")

    def add_indicator(self, name: str, indicator: Indicator, column: str = 'Close'):
        """
        Add a new indicator to the backtesting environment.

        Args:
            name (str): Name of the indicator.
            indicator (Indicator): Indicator object.
            column (str, optional): Data column to apply the indicator to. Defaults to 'Close'.
        """
        self.indicator_calculator.add_indicator(name, indicator, column)
        
    def log_trade(self, date, symbol, action, price, size, change):
        """
        Log details of a trade for analysis.

        Args:
            date (datetime.datetime): Date of the trade.
            symbol (str): Symbol traded.
            action (str): Buy or Sell.
            price (float): Execution price.
            size (int): Number of units traded.
            change (float): Change in cash balance due to the trade.
        """
        logging.info(f"TRADE EXECUTED - Date: {date}, Symbol: {symbol}, Action: {action}, Price: {price}, Size: {size}, Change from execution: {change}")

    def run_backtest(self) -> List[float]:
        """
        Run the backtest simulation.

        Returns:
            List[float]: List of daily portfolio values throughout the backtest.
        """
        logging.debug("Starting backtest")
        
        if self.strategy is None:
            logging.error("Strategy not set. Please set a strategy before running the backtest.")
            raise ValueError("Strategy not set. Please set a strategy before running the backtest.")

        min_start = self.indicator_calculator.calculate_indicators()
        portfolio_values = []
        daily_cash = []
        dates = list(self.date_to_index.keys())
        self.trade_history = []
        
        # Main backtest loop
        for date_idx in range(min_start, self.data.shape[0] - 1):
            date = dates[date_idx]
            daily_data = self.data[date_idx]
            
            # Generate trading signals for the current day
            self.strategy.generate_signals(date_idx, self.indicator_calculator, self.positions, self.symbol_to_index)
            signals = self.strategy.get_signals()
            
            # Execute pending orders from the previous day
            self._execute_pending_orders(date, daily_data)
            
            # Calculate and record portfolio value and cash balance
            portfolio_value = self.calculate_portfolio_value(daily_data=daily_data)
            portfolio_values.append(portfolio_value)
            daily_cash.append(self.cash)

            # Process new signals and create orders
            self._process_signals(date, signals, daily_data)

        logging.debug("Backtest completed")
        self.generate_backtest_report(np.array(portfolio_values), dates[min_start:], [], daily_cash, self.trade_history)
        return portfolio_values

    def _execute_pending_orders(self, date, daily_data):
        """
        Execute all pending orders for the current day.

        Args:
            date (datetime.datetime): Current date.
            daily_data (np.ndarray): Market data for the current day.
        """
        for sym in self.orders:
            while self.orders[sym]:
                order = self.orders[sym][0]
                change, qty, execution_price, action = order.execute(daily_data, self.feature_to_index, date)
                self.cash += change
                self.log_trade(date, sym, action, execution_price, qty, change)
                self.trade_history.append({
                    'date': date,
                    'symbol': sym,
                    'action': action,
                    'price': execution_price,
                    'size': qty,
                    'cash (after)': self.cash,
                })
                self.orders[sym].pop(0)

    def _process_signals(self, date, signals, daily_data):
        """
        Process trading signals and create new orders.

        Args:
            date (datetime.datetime): Current date.
            signals (Dict[str, int]): Trading signals for each symbol.
            daily_data (np.ndarray): Market data for the current day.
        """
        for symbol, signal in signals.items():
            symbol_idx = self.symbol_to_index[symbol]
            price = daily_data[symbol_idx, self.feature_to_index['Close']]

            if np.isnan(price) or price == 0:
                price = self._handle_invalid_price(symbol, date, price)
                if price is None:
                    continue

            position = self.positions[symbol]

            if signal == Strategy.BUY and self.cash >= price * self.trade_size:
                self._create_buy_order(symbol, date, price, position, symbol_idx)
            elif signal == Strategy.SELL and position.get_current_quantity() >= self.trade_size:
                self._create_sell_order(symbol, date, price, position, symbol_idx)

            self._handle_position_with_invalid_price(symbol, position, price, date, symbol_idx)

    def _handle_invalid_price(self, symbol, date, price):
        """
        Handle cases where the price is invalid (NaN or zero).

        Args:
            symbol (str): Symbol being traded.
            date (datetime.datetime): Current date.
            price (float): Current price (which may be invalid).

        Returns:
            float: Valid price to use, or None if no valid price is available.
        """
        if self.last_valid_price[symbol] is not None:
            price = self.last_valid_price[symbol]
            logging.warning(f"Invalid price for {symbol} on {date}. Using last valid price: {price}")
        else:
            logging.error(f"No valid price available for {symbol} on {date}. Skipping trade.")
            return None
        return price

    def _create_buy_order(self, symbol, date, price, position, symbol_idx):
        """
        Create a buy order.

        Args:
            symbol (str): Symbol to buy.
            date (datetime.datetime): Current date.
            price (float): Current price.
            position (Position): Current position for the symbol.
            symbol_idx (int): Index of the symbol in the data array.
        """
        logging.info(f'TRADE - ORDER CREATED, Date: {date} BUY MKT at {price} on {symbol} with {self.trade_size} units. {self.cash}/{self.calculate_portfolio_value(self.data[self.date_to_index[date]])}')
        self.orders[symbol].append(Order(Strategy.BUY, date, self.trade_size, position, symbol_idx))

    def _create_sell_order(self, symbol, date, price, position, symbol_idx):
        """
        Create a sell order.

        Args:
            symbol (str): Symbol to sell.
            date (datetime.datetime): Current date.
            price (float): Current price.
            position (Position): Current position for the symbol.
            symbol_idx (int): Index of the symbol in the data array.
        """
        logging.info(f'TRADE - ORDER CREATED, Date: {date} SELL MKT at {price} on {symbol} with {self.trade_size} units. {self.cash}/{self.calculate_portfolio_value(self.data[self.date_to_index[date]])}')
        self.orders[symbol].append(Order(Strategy.SELL, date, self.trade_size, position, symbol_idx))

    def _handle_position_with_invalid_price(self, symbol, position, price, date, symbol_idx):
        """
        Handle cases where we have a position but the current price is invalid.

        Args:
            symbol (str): Symbol being traded.
            position (Position): Current position for the symbol.
            price (float): Current price (which may be invalid).
            date (datetime.datetime): Current date.
            symbol_idx (int): Index of the symbol in the data array.
        """
        if position.get_current_quantity() > 0 and (np.isnan(price) or price == 0):
            logging.warning(f"Invalid price for {symbol} while holding position. Selling at last valid price.")
            sell_price = self.last_valid_price[symbol]
            self.orders[symbol].append(Order(Strategy.SELL, date, self.trade_size, position, symbol_idx))

    def calculate_portfolio_value(self, daily_data: np.ndarray) -> float:
        """
        Calculate the total portfolio value.

        Args:
            daily_data (np.ndarray): Market data for the current day.

        Returns:
            float: Total portfolio value (cash + positions).
        """
        total_value = self.cash + sum(
            position.calculate_value(daily_data[self.symbol_to_index[symbol], self.feature_to_index['Close']])
            for symbol, position in self.positions.items()
        )
        return total_value

    def get_state(self):
        """
        Get the current state of the backtesting environment.

        Returns:
            dict: Current state including cash, positions, and indicator values.
        """
        logging.debug("Getting current state of Strato")
        state = {
            'cash': self.cash,
            'positions': {symbol: position.get_positions() for symbol, position in self.positions.items()},
            'indicators': {name: values.tolist() for name, values in self.indicator_calculator.indicator_values.items()}
        }
        logging.debug(f"Current state: {state}")
        return state
    
    def _calculate_upi(self, returns, risk_free_rate=0.02):
        """
        Calculate the Ulcer Performance Index.

        Args:
            returns (np.array): Array of returns.
            risk_free_rate (float, optional): Risk-free rate. Defaults to 0.02.

        Returns:
            float: Ulcer Performance Index.
        """
        drawdowns = 1 - returns / np.maximum.accumulate(returns)
        ulcer_index = np.sqrt(np.mean(np.square(drawdowns)))
        upi = (np.mean(returns) - risk_free_rate) / ulcer_index
        return upi
    
    def _calculate_sharpe_ratio(self, returns, annual_risk_free_rate=0.01, trading_days=252, convert_rate=True, annualize=False, stddev_sample=False):
        """
        Calculate the Sharpe ratio for a given set of returns.

        Args:
            returns (np.ndarray): Array of daily returns.
            annual_risk_free_rate (float): Annual risk-free rate. Defaults to 0.02 (2%).
            trading_days (int): Number of trading days in a year. Defaults to 252.
            convert_rate (bool): Whether to convert the annual risk-free rate to the daily rate. Defaults to True.
            annualize (bool): Whether to annualize the Sharpe ratio. Defaults to False.
            stddev_sample (bool): Whether to apply Bessel's correction. Defaults to False.

        Returns:
            float: The calculated Sharpe ratio.
        """
        if convert_rate:
            daily_risk_free_rate = (1 + annual_risk_free_rate) ** (1 / trading_days) - 1
        else:
            daily_risk_free_rate = annual_risk_free_rate
        
        excess_returns = returns - daily_risk_free_rate
        mean_excess_return = np.mean(excess_returns) * trading_days
        
        if stddev_sample:
            std_excess_return = np.std(excess_returns, ddof=1) * np.sqrt(trading_days)
        else:
            std_excess_return = np.std(excess_returns) * np.sqrt(trading_days)

        if std_excess_return == 0:
            return 0  # Avoid division by zero
        
        sharpe_ratio = mean_excess_return / std_excess_return
        
        if annualize and convert_rate:
            sharpe_ratio *= np.sqrt(trading_days)
        
        return sharpe_ratio

    def generate_backtest_report(self, portfolio_values: np.ndarray, dates, daily_pnl, daily_cash, trade_history):
        """
        Generate a comprehensive backtest report.

        This method creates various plots and statistics based on the backtest results.
        The report is saved as a LaTeX document which can be compiled into a PDF.

        Args:
            portfolio_values (List[float]): Daily portfolio values.
            dates (List[datetime.datetime]): Corresponding dates.
            daily_pnl (List[float]): Daily profit and loss.
            daily_cash (List[float]): Daily cash balances.
            trade_history (List[dict]): History of all trades executed.

        Returns:
            str: Path to the generated LaTeX file.
        """
        logging.info("Generating backtest report")
        
        # Create a temporary directory to save plots
        temp_dir = tempfile.mkdtemp()

        try:
            # Calculate daily returns
            returns = np.array(portfolio_values[1:] / portfolio_values[:-1] - 1)
            
            # Calculate Sharpe ratio
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            
            # Calculate cumulative returns
            cumulative_returns = self._calculate_cumulative_returns(returns)
            
            # Calculate logarithmic returns
            log_returns = np.log(portfolio_values[1:] / portfolio_values[:-1])
            
            # Calculate maximum drawdown
            max_drawdown = self._calculate_max_drawdown(portfolio_values)
            
            # Calculate normalized annual return
            normalized_annual_return = self._calculate_normalized_annual_return(returns)
            
            # Save plots
            portfolio_value_path = os.path.join(temp_dir, 'portfolio_value.png')
            self._plot_portfolio_value(dates, portfolio_values, portfolio_value_path)

            cumulative_returns_path = os.path.join(temp_dir, 'cumulative_returns.png')
            self._plot_cumulative_returns(dates[1:], cumulative_returns, cumulative_returns_path)

            log_returns_path = os.path.join(temp_dir, 'log_returns.png')
            self._plot_logarithmic_returns(dates[1:], log_returns, log_returns_path)
            
            returns_quantile_6m_path = os.path.join(temp_dir, 'returns_quantile_6m.png')
            self._plot_returns_quantile(dates[1:], log_returns, 6, returns_quantile_6m_path)
            
            returns_quantile_12m_path = os.path.join(temp_dir, 'returns_quantile_12m.png')
            self._plot_returns_quantile(dates[1:], log_returns, 12, returns_quantile_12m_path)
            
            returns_distribution_path = os.path.join(temp_dir, 'returns_distribution.png')
            self._plot_returns_distribution(returns, returns_distribution_path)

            monthly_returns_heatmap_path = os.path.join(temp_dir, 'monthly_returns_heatmap.png')
            self._plot_monthly_returns_heatmap(dates[1:], log_returns, monthly_returns_heatmap_path)

            # Generate LaTeX document
            latex_file_path = os.path.join(temp_dir, 'backtest_report.tex')
            self._generate_latex_document(
                latex_file_path,
                portfolio_value_path,
                cumulative_returns_path,
                log_returns_path,
                returns_quantile_6m_path,
                returns_quantile_12m_path,
                returns_distribution_path,
                monthly_returns_heatmap_path,
                sharpe_ratio,
                max_drawdown,
                normalized_annual_return
            )

            # Compile LaTeX to PDF
            subprocess.run(['pdflatex', '-output-directory', temp_dir, latex_file_path], check=True)

            # Move PDF to current working directory
            pdf_file_name = 'backtest_report.pdf'
            pdf_file_path = os.path.join(temp_dir, pdf_file_name)
            final_pdf_path = os.path.join(os.getcwd(), pdf_file_name)
            shutil.move(pdf_file_path, final_pdf_path)

            return final_pdf_path

        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)

    def _generate_latex_document(self, filepath, portfolio_value_path, cumulative_returns_path, log_returns_path,
                                returns_quantile_6m_path, returns_quantile_12m_path, returns_distribution_path,
                                monthly_returns_heatmap_path, sharpe_ratio, max_drawdown, normalized_annual_return):
        """
        Generate a LaTeX document for the backtest report.

        Args:
            filepath (str): Path to save the LaTeX file.
            portfolio_value_path (str): Path to the portfolio value plot.
            cumulative_returns_path (str): Path to the cumulative returns plot.
            log_returns_path (str): Path to the log returns plot.
            returns_quantile_6m_path (str): Path to the 6-month returns quantile plot.
            returns_quantile_12m_path (str): Path to the 12-month returns quantile plot.
            returns_distribution_path (str): Path to the returns distribution plot.
            monthly_returns_heatmap_path (str): Path to the monthly returns heatmap plot.
            sharpe_ratio (float): Sharpe ratio.
            max_drawdown (float): Maximum drawdown.
            normalized_annual_return (float): Normalized annual return.
        """
        latex_content = r"""
        \documentclass{article}
        \usepackage{graphicx}
        \usepackage{geometry}
        \geometry{a4paper, margin=1in}
        \begin{document}
        \title{Backtest Report}
        \author{}
        \date{}
        \maketitle

        \section*{Summary Statistics}
        \begin{itemize}
        \item Sharpe Ratio: %.2f
        \item Maximum Drawdown: %.2f %%
        \item Normalized Annual Return: %.2f %%
        \end{itemize}

        \section*{Plots}

        \begin{figure}[h!]
        \centering
        \includegraphics[width=0.32\textwidth]{%s}
        \includegraphics[width=0.32\textwidth]{%s}
        \includegraphics[width=0.32\textwidth]{%s}
        \caption{Portfolio Value, Cumulative Returns, and Log Returns}
        \end{figure}

        \begin{figure}[h!]
        \centering
        \includegraphics[width=0.45\textwidth]{%s}
        \includegraphics[width=0.45\textwidth]{%s}
        \caption{6-Month and 12-Month Returns Quantile}
        \end{figure}

        \begin{figure}[h!]
        \centering
        \includegraphics[width=0.45\textwidth]{%s}
        \includegraphics[width=0.45\textwidth]{%s}
        \caption{Returns Distribution and Monthly Returns Heatmap}
        \end{figure}

        \end{document}
        """ % (sharpe_ratio, max_drawdown, normalized_annual_return, portfolio_value_path, cumulative_returns_path, log_returns_path, returns_quantile_6m_path, returns_quantile_12m_path, returns_distribution_path, monthly_returns_heatmap_path)

        with open(filepath, 'w') as f:
            f.write(latex_content)


    def _plot_portfolio_value(self, dates, portfolio_values, save_path):
        """
        Plot the portfolio value over time.

        Args:
            dates (List[datetime.datetime]): Corresponding dates.
            portfolio_values (np.array): Array of portfolio values.
            save_path (str): Path to save the plot.
        """
        plt.figure(figsize=(14, 7))
        plt.plot(dates[:-1], portfolio_values, label='Portfolio Value')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value (₹)')
        plt.title('Portfolio Value Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def _plot_cumulative_returns(self, dates, cumulative_returns, save_path):
        """
        Plot the cumulative returns over time.

        Args:
            dates (List[datetime.datetime]): Corresponding dates.
            cumulative_returns (np.array): Cumulative returns.
            save_path (str): Path to save the plot.
        """
        plt.figure(figsize=(14, 7))
        plt.plot(dates[:-1], cumulative_returns, label='Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.title('Cumulative Returns Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def _plot_logarithmic_returns(self, dates, log_returns, save_path):
        """
        Plot the logarithmic returns over time.

        Args:
            dates (List[datetime.datetime]): Corresponding dates.
            log_returns (np.array): Logarithmic returns.
            save_path (str): Path to save the plot.
        """
        plt.figure(figsize=(14, 7))
        plt.plot(dates[:-1], log_returns, label='Logarithmic Returns')
        plt.xlabel('Date')
        plt.ylabel('Logarithmic Returns')
        plt.title('Logarithmic Returns Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def _plot_returns_quantile(self, dates, returns, months, save_path):
        """
        Plot the returns quantile over a specified period.

        Args:
            dates (List[datetime.datetime]): Corresponding dates.
            returns (np.array): Array of daily returns.
            months (int): Number of months for the quantile calculation.
            save_path (str): Path to save the plot.
        """
        period_returns = [np.prod(1 + returns[i:i+months*21]) - 1 for i in range(len(returns) - months*21)]
        quantiles = np.percentile(period_returns, [5, 25, 50, 75, 95])
        
        plt.figure(figsize=(14, 7))
        plt.plot(range(len(quantiles)), quantiles, marker='o')
        plt.xlabel('Quantile')
        plt.ylabel('Returns')
        plt.title(f'{months}-Month Returns Quantile')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def _plot_returns_distribution(self, returns, save_path):
        """
        Plot the distribution of returns.

        Args:
            returns (np.array): Array of daily returns.
            save_path (str): Path to save the plot.
        """
        plt.figure(figsize=(14, 7))
        plt.hist(returns, bins=50, alpha=0.75)
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.title('Returns Distribution')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def _plot_monthly_returns_heatmap(self, dates, returns, save_path):
        """
        Plot a heatmap of monthly returns.

        Args:
            dates (List[datetime.datetime]): Corresponding dates.
            returns (np.array): Array of daily returns.
            save_path (str): Path to save the plot.
        """
        import pandas as pd
        import calendar

        # Convert dates to a DataFrame to extract month and year
        df = pd.DataFrame({'date': dates[1:], 'returns': returns})
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

        # Calculate monthly returns
        monthly_returns = df.groupby(['year', 'month'])['returns'].apply(lambda x: (x + 1).prod() - 1).unstack()

        # Create a heatmap
        plt.figure(figsize=(14, 7))
        sns.heatmap(monthly_returns, annot=True, fmt=".2%", cmap='coolwarm', cbar=True, center=0, linewidths=.5)
        plt.title('Monthly Returns Heatmap')
        plt.xlabel('Month')
        plt.ylabel('Year')
        plt.yticks(rotation=0)
        plt.xticks(ticks=np.arange(0.5, 12.5), labels=[calendar.month_abbr[i] for i in range(1, 13)], rotation=0)
        plt.savefig(save_path)
        plt.close()


    def _calculate_cumulative_returns(self, returns):
        """
        Calculate the cumulative returns.

        Args:
            returns (np.array): Array of daily returns.

        Returns:
            np.array: Cumulative returns.
        """
        cumulative_returns = np.cumprod(1 + returns) - 1
        return cumulative_returns

    def _calculate_max_drawdown(self, portfolio_values):
        """
        Calculate the maximum drawdown.

        Args:
            portfolio_values (np.array): Array of portfolio values.

        Returns:
            float: Maximum drawdown.
        """
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        return max_drawdown * 100

    def _calculate_normalized_annual_return(self, returns, trading_days=252):
        """
        Calculate the normalized annual return.

        Args:
            returns (np.array): Array of daily returns.
            trading_days (int): Number of trading days in a year. Defaults to 252.

        Returns:
            float: Normalized annual return.
        """
        annual_return = np.prod(1 + returns) ** (trading_days / len(returns)) - 1
        return annual_return * 100

