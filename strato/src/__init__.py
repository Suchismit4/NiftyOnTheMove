# Standard library imports
import datetime
import logging
import os
from typing import Dict, List
import tempfile
import subprocess
import calendar
import shutil

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from backtrader import strategies
from numba.core.utils import benchmark

from .flow.broker import Broker

plt.style.use('dark_background')
from pandas import Timestamp
import seaborn as sns
import matplotlib.dates as mdates

# Local imports
from .struct.strategy import Strategy
from .flow.position import Position
from .struct.indicator import IndicatorCalculator, Indicator
from .flow.order import Order

def log_trade(date, symbol, action, price, size, change):
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


class Strato:
    """
    Main class for backtesting trading strategies.
    
    This class handles the simulation of trading, including position management,
    order execution, and performance tracking.
    """
    
    def __init__(self, data: np.ndarray, 
                 symbol_to_index: Dict[str, int], 
                 feature_to_index: Dict[str, int], 
                 date_to_index: Dict[datetime.datetime, int], 
                 starting_cash: float, 
                 strategies: List[Strategy], 
                 benchmark: pd.DataFrame = None, 
                 generate_report: bool = False,
                 name: str = None):
        """
        Initialize the Strato backtesting environment.

        Args:
            data (np.ndarray): Historical market data.
            symbol_to_index (Dict[str, int]): Mapping of symbols to their indices in the data array.
            feature_to_index (Dict[str, int]): Mapping of features to their indices in the data array.
            date_to_index (Dict[datetime.datetime, int]): Mapping of dates to their indices in the data array.
            starting_cash (float): Initial cash balance.
            strategies (List[Strategy]): Trading strategies to be tested.
            benchmark (pd.DataFrame): OPTIONAL If any benchmark is to be compared.
            generate_report (bool): OPTIONAL specifies if we need to export a detailed report.
        """
        self.trade_history = None
        logging.debug("Initializing Strato class")
        
        self.data = data
        self.symbol_to_index = symbol_to_index
        self.feature_to_index = feature_to_index
        self.date_to_index = date_to_index
        self.starting_cash = starting_cash
        self.strategies = strategies
        self.name = name
        
        # Initialize positions, cash, and orders
        self.positions = {symbol: Position(symbol, 0, 0, datetime.datetime.now())
                          for symbol in symbol_to_index.keys()}

        self.cash = starting_cash
        self.orders = {symbol: [] for symbol in symbol_to_index.keys()}
        
        # Initialize indicator calculator and last valid prices to track last valid prices
        self.indicator_calculator = IndicatorCalculator(data, feature_to_index)
        self.last_valid_price = {symbol: None for symbol in symbol_to_index.keys()}
        
        logging.debug(f"Initialized with data shape: {data.shape}, starting cash: {starting_cash}")
        logging.debug(f"Symbols: {list(symbol_to_index.keys())}, Features: {list(feature_to_index.keys())}")
        self.benchmark = benchmark
        
        self.generate_report = generate_report
        
        self.broker = Broker()
        self.broker.set(self.cash, self.cash)


    def add_indicator(self, name: str, indicator: Indicator):
        """
        Add a new indicator to the backtesting environment.

        Args:
            name (str): Name of the indicator.
            indicator (Indicator): Indicator object.
        """
        self.indicator_calculator.add_indicator(name, indicator)

    def run_backtest(self, start_date: str = None) -> List[float]:
        """
        Run the backtest simulation.

        Returns:
            List[float]: List of daily portfolio values throughout the backtest.
        """
        logging.debug("Starting backtest")
        
        if self.strategies is None:
            logging.error("Strategy not set. Please set a strategy before running the backtest.")
            raise ValueError("Strategy not set. Please set a strategy before running the backtest.")

        min_start = self.indicator_calculator.calculate_indicators()
        portfolio_values = []
        daily_cash = []
        dates = list(self.date_to_index.keys())

        # Set the broker to all strategy objects
        for strategy in self.strategies:
            strategy.set_broker(self.broker)

        self.trade_history = []

        if start_date is not None:
            try:
                # Convert start_date to Timestamp if it's not already
                start_timestamp = Timestamp(start_date)
                
                # Find the index corresponding to the start date
                min_start = next(idx for date, idx in self.date_to_index.items() if date == start_timestamp)
            except StopIteration:
                raise Exception("Invalid Date to start with")
        
        # Main backtest loop
        for date_idx in range(min_start, self.data.shape[0] - 1):
            date = dates[date_idx]
            daily_data = self.data[date_idx]

            signals = {s.name: {} for s in self.strategies}

            # Execute pending orders from the previous day
            self._execute_pending_orders(date, daily_data)

            # Loop through strategies in priority executing signals
            # Generate trading signals for the current day
            for strategy in self.strategies:

                strategy.step(date_idx, self.indicator_calculator, self.positions, self.symbol_to_index, benchmark=self.benchmark)
                signals[strategy.name] = strategy.get_signals()

            # Calculate and record portfolio value and cash balance
            portfolio_value = self.calculate_portfolio_value(daily_data=daily_data)
            portfolio_values.append(portfolio_value)
            self.broker.set(self.cash, portfolio_value)
            daily_cash.append(self.cash)

            # Process new signals and create orders
            for signal in signals.values():
                self._process_signals(date, signal, daily_data)

            for symbol in self.symbol_to_index.keys():
                self._handle_invalid_price(symbol=symbol, date=date, price=daily_data[self.symbol_to_index[symbol], self.feature_to_index['Open']])
            
            for symbol in self.symbol_to_index.keys():
                self._handle_position_with_invalid_price(symbol, self.positions[symbol], daily_data[self.symbol_to_index[symbol], self.feature_to_index['Open']], date, self.symbol_to_index[symbol])
            
            self._update_positions()

        logging.debug("Backtest completed")
        if self.generate_report:
            self.generate_backtest_report(np.array(portfolio_values), dates[min_start:], [], daily_cash, self.trade_history)
        return portfolio_values

    def _update_positions(self):

        for sym in self.positions:
            pos = self.positions[sym]
            
            if len(pos.get_open_lots()) >= 0:
                pos.bars_since_entry += 1 # fix this to reset

    def _execute_pending_orders(self, date, daily_data):
        """
        Execute all pending orders for the current day from previous day.

        Args:
            date (datetime.datetime): Current date.
            daily_data (np.ndarray): Market data for the current day.
        """
        sell_orders = []
        buy_orders = []

        # Collect all orders across all symbols
        for sym in self.orders:
            for order in self.orders[sym]:
                if order.order_type == Strategy.SELL and (sym, order) not in sell_orders:
                    sell_orders.append((sym, order))
                elif order.order_type == Strategy.BUY and (sym, order) not in buy_orders:
                    buy_orders.append((sym, order))

        # Process all sell orders first
        for sym, order in sell_orders:
            change, qty, execution_price, action, realized_pnl, closed_lots = order.execute(daily_data, self.feature_to_index, date)
            self.cash += change

            for closed_lot in closed_lots:
                self.trade_history.append({
                    'type': 'TRADE_CLOSED',
                    'symbol': sym,
                    'buy_date': closed_lot['buy_date'],
                    'sell_date': closed_lot['sell_date'],
                    'quantity': closed_lot['quantity'],
                    'buy_price': closed_lot['buy_price'],
                    'sell_price': closed_lot['sell_price'],
                    'pnl': closed_lot['pnl'],
                    'cash_after': self.cash
                })
            self.orders[sym].remove(order)

            logging.info(f"TRADE - ORDER EXECUTED - {date} for {sym} SELL MKT @{execution_price}::{qty}")


        # Process all buy orders next
        for sym, order in buy_orders:
            change, qty, execution_price, action, _, _ = order.execute(daily_data, self.feature_to_index, date)
            self.cash += change

            self.trade_history.append({
                'type': 'ACTION',
                'date': date,
                'symbol': sym,
                'action': action,
                'price': execution_price,
                'size': qty,
                'cash_after': self.cash,
            })
            self.orders[sym].remove(order)

            logging.info(f"TRADE - ORDER EXECUTED - {date} for {sym} BUY MKT @{execution_price}::{qty}")

    def _process_signals(self, date, signals, daily_data):
        """
        Process trading signals and create new orders.

        Args:
            date (datetime.datetime): Current date.
            signals (Dict[str, int]): Trading signals for each symbol.
            daily_data (np.ndarray): Market data for the current day.
        """
        disposable_cash = self.cash
        
        for symbol, order in signals.items():
            signal = order[0]
            quantity = order[1]
            symbol_idx = self.symbol_to_index[symbol]
            price = daily_data[symbol_idx, self.feature_to_index['Close']]

            if np.isnan(price) or price == 0:
                price = self._handle_invalid_price(symbol, date, price)
                if price is None:
                    continue

            position = self.positions[symbol]

            if signal == Strategy.BUY:
                if disposable_cash >= self.data[self.date_to_index[date] + 1, symbol_idx, self.feature_to_index['Open']] * quantity:
                    disposable_cash -= self.data[self.date_to_index[date] + 1, symbol_idx, self.feature_to_index['Open']] * quantity
                    self._create_buy_order(symbol, date, price, position, symbol_idx, quantity)
                else:
                    logging.error(f"Tried to enter on {date} in {symbol} with {position} units for price of {price}. Rejected because of insufficient liquid funds: {disposable_cash}")
            elif signal == Strategy.SELL and position.get_current_quantity() >= quantity:
                self._create_sell_order(symbol, date, price, position, symbol_idx, quantity)

    def _handle_invalid_price(self, symbol, date, price: np.ndarray):
        """
        Handle cases where the price is invalid (NaN or zero).

        Args:
            symbol (str): Symbol being traded.
            date (datetime.datetime): Current date.
            price (np.ndarray): Current price (which may be invalid).
        """
        if self.last_valid_price[symbol] is None and (not np.isnan(price) or price != 0.):
            self.last_valid_price[symbol] = price
            
    def _create_buy_order(self, symbol, date, price, position, symbol_idx, quantity):
        """
        Create a buy order.

        Args:
            symbol (str): Symbol to buy.
            date (int): Current date.
            price (float): Current price.
            position (Position): Current position for the symbol.
            symbol_idx (int): Index of the symbol in the data array.
        """
        logging.info(f'TRADE - ORDER CREATED, Date: {date} BUY MKT at {price} on {symbol} with {quantity} units. {self.cash}/{self.calculate_portfolio_value(self.data[self.date_to_index[date]])}')
        self.orders[symbol].append(Order(Strategy.BUY, date, quantity, position, symbol_idx))

    def _create_sell_order(self, symbol, date, price, position, symbol_idx, quantity):
        """
        Create a sell order.

        Args:
            symbol (str): Symbol to sell.
            date (int): Current date.
            price (float): Current price.
            position (Position): Current position for the symbol.
            symbol_idx (int): Index of the symbol in the data array.
        """
        logging.info(f'TRADE - ORDER CREATED, Date: {date} SELL MKT at {price} on {symbol} with {quantity} units. {self.cash}/{self.calculate_portfolio_value(self.data[self.date_to_index[date]])}')
        self.orders[symbol].append(Order(Strategy.SELL, date, quantity, position, symbol_idx))

    def _handle_position_with_invalid_price(self, symbol, position, price, date, symbol_idx):
        """
        Handle cases where we have a position but the current price is invalid.

        Args:
            symbol (str): Symbol being traded.
            position (Position): Current position for the symbol.
            price (np.ndarray): Current price (which may be invalid).
            date (int): Current date.
            symbol_idx (int): Index of the symbol in the data array.
        """
        if position.get_current_quantity() > 0 and (np.isnan(price) or price == 0):
            logging.error(f"Invalid price for {symbol} while holding position. Selling at last valid price.")
            sell_price = self.last_valid_price[symbol]
            if not np.isnan(sell_price) or sell_price != 0.:
                self.orders[symbol].append(Order(Strategy.SELL, date, position.get_current_quantity(), position, symbol_idx, execution_price_force=sell_price))
            else:
                logging.error(f'MISSING PRICE - Tried to create a sell order for a suddenly missing symbol {symbol} but failed at date {date} for {position.get_current_quantity()}')

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
            'positions': {symbol: position.get_open_lots() for symbol, position in self.positions.items()},
            'indicators': {name: values.tolist() for name, values in self.indicator_calculator.indicator_values.items()}
        }
        logging.debug(f"Current state: {state}")
        return state
    
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

            returns_quantile_3m_path = os.path.join(temp_dir, 'returns_quantile_3m.png')
            self._plot_returns_quantile(dates[1:], returns, 3, returns_quantile_3m_path)
            
            returns_quantile_6m_path = os.path.join(temp_dir, 'returns_quantile_6m.png')
            self._plot_returns_quantile(dates[1:], returns, 6, returns_quantile_6m_path)
            
            returns_quantile_12m_path = os.path.join(temp_dir, 'returns_quantile_12m.png')
            self._plot_returns_quantile(dates[1:], returns, 12, returns_quantile_12m_path)
            
            returns_distribution_path = os.path.join(temp_dir, 'returns_distribution.png')
            self._plot_returns_distribution(returns, returns_distribution_path)

            monthly_returns_heatmap_path = os.path.join(temp_dir, 'monthly_returns_heatmap.png')
            self._plot_monthly_returns_heatmap(dates[1:], returns, monthly_returns_heatmap_path)

            yearly_returns_path = os.path.join(temp_dir, 'yearly_returns.png')
            self._plot_yearly_returns(dates[1:], returns, yearly_returns_path)

            rolling_sharpe_path = os.path.join(temp_dir, 'rolling_sharpe.png')
            self._plot_rolling_sharpe_ratio(dates[1:], returns, rolling_sharpe_path, 126)

            rolling_volatility_path = os.path.join(temp_dir, 'rolling_volatility.png')
            self._plot_rolling_volatility(dates[1:], returns,rolling_volatility_path, 126)
            
            weekly_returns_path = os.path.join(temp_dir, 'weekly_returns.png')
            self._plot_weekly_returns(dates[1:], returns, weekly_returns_path)
            
            drawdowns_path = os.path.join(temp_dir, 'drawdowns.png')
            self._plot_drawdowns(dates, portfolio_values, drawdowns_path)

            trade_analysis = self._analyze_trades()

            # Generate LaTeX document
            latex_file_path = os.path.join(temp_dir, 'backtest_report.tex')
            self._generate_latex_document(
                latex_file_path,
                portfolio_value_path,
                cumulative_returns_path,
                returns_quantile_3m_path,
                returns_quantile_6m_path,
                returns_quantile_12m_path,
                returns_distribution_path,
                monthly_returns_heatmap_path,
                yearly_returns_path,
                rolling_sharpe_path,
                rolling_volatility_path,
                sharpe_ratio,
                max_drawdown,
                normalized_annual_return,
                weekly_returns_path,
                drawdowns_path,
                all_total_trades=trade_analysis['all_trades']['total_trades'],
                all_trades_per_year=trade_analysis['all_trades']['trades_per_year'],
                all_average_return=trade_analysis['all_trades']['average_return_per_trade'],
                winning_total_trades=trade_analysis['winning_trades']['total_trades'],
                winning_trades_per_year=trade_analysis['winning_trades']['trades_per_year'],
                winning_average_return=trade_analysis['winning_trades']['average_return_per_trade'],
                winning_best_trade=trade_analysis['winning_trades']['best_trade'],
                winning_max_duration=trade_analysis['winning_trades']['max_duration'],
                winning_min_duration=trade_analysis['winning_trades']['min_duration'],
                losing_total_trades=trade_analysis['losing_trades']['total_trades'],
                losing_trades_per_year=trade_analysis['losing_trades']['trades_per_year'],
                losing_average_return=trade_analysis['losing_trades']['average_return_per_trade'],
                losing_worst_trade=trade_analysis['losing_trades']['worst_trade'],
                losing_max_duration=trade_analysis['losing_trades']['max_duration'],
                losing_min_duration=trade_analysis['losing_trades']['min_duration'],
                win_rate=trade_analysis['win_rate'] * 100,
                lose_rate=trade_analysis['lose_rate'] * 100,
                profit_factor=trade_analysis['profit_factor'],
                win_loss_ratio=trade_analysis['win_loss_ratio'],
                payoff_ratio=trade_analysis['payoff_ratio'],
                cpc_index=trade_analysis['cpc_index'],
                expectancy=trade_analysis['expectancy']
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

    def _generate_latex_document(self, filepath, portfolio_value_path, cumulative_returns_path,
                                 returns_quantile_3m_path, returns_quantile_6m_path, returns_quantile_12m_path,
                                 returns_distribution_path, monthly_returns_heatmap_path,
                                 yearly_returns_path, rolling_sharpe_path, rolling_volatility_path,
                                 sharpe_ratio, max_drawdown, normalized_annual_return, weekly_returns_path,
                                 drawdowns_path, all_total_trades, all_trades_per_year, all_average_return,
                                 winning_total_trades, winning_trades_per_year, winning_average_return,
                                 winning_best_trade, winning_max_duration, winning_min_duration,
                                 losing_total_trades, losing_trades_per_year, losing_average_return,
                                 losing_worst_trade, losing_max_duration, losing_min_duration,
                                 win_rate, lose_rate, profit_factor, win_loss_ratio, payoff_ratio,
                                 cpc_index, expectancy):
        latex_content = r"""
        \documentclass{article}
        \usepackage{graphicx}
        \usepackage{xcolor}
        \pagecolor[rgb]{0,0,0} 
        \color[rgb]{0.96078431372, 0.96470588235, 0.98039215686} 
        \usepackage{geometry}
        \usepackage{subfig}
        \geometry{a4paper, margin=0.8in}
        \begin{document}
        \title{%s}
        \author{}
        \date{}
        \maketitle

        \section*{Summary Statistics}
        \begin{tabular}{|l|r|}
        \hline
        Metric & Value \\
        \hline
        Sharpe Ratio & %.2f \\
        Maximum Drawdown [per] & %.2f\\
        Normalized Annual Return  [per] & %.2f  \\
        \hline
        \end{tabular}

        \section*{Trade Analysis (Experimental)}
        \begin{tabular}{|l|l|r|}
        \hline
        & Metric & Value \\
        \hline
        \textbf{All Trades} & Total Trades & %d \\
        & Trades per Year & %.2f \\
        & Average Return per Trade [per] & %.2f \\
        \hline
        \textbf{Winning Trades} & Total Trades & %d \\
        & Trades per Year & %.2f \\
        & Average Return per Trade [per] & %.2f \\
        & Best Trade [per] & %.2f \\
        & Maximum Duration & %d days \\
        & Minimum Duration & %d days \\
        \hline
        \textbf{Losing Trades} & Total Trades & %d \\
        & Trades per Year & %.2f \\
        & Average Return per Trade [per] & %.2f \\
        & Worst Trade [per] & %.2f \\
        & Maximum Duration & %d days \\
        & Minimum Duration & %d days \\
        \hline
        \textbf{Overall Metrics} & Win Rate [per] & %.2f \\
        & Lose Rate [per] & %.2f \\
        & Profit Factor & %.2f \\
        & Win/Loss Ratio & %.2f \\
        & Payoff Ratio & %.2f \\
        & CPC Index & %.2f \\
        & Expectancy & %.2f \\
        \hline
        \end{tabular}


        \section*{Performance}
        \begin{figure}[h!]
        \centering
        \subfloat[Cumulative Returns]{\includegraphics[width=0.48\textwidth]{%s}}
        \hfill
        \subfloat[Portfolio Value]{\includegraphics[width=0.48\textwidth]{%s}}
        \caption{Cumulative Returns and Portfolio Value Over Time}
        \end{figure}

        \section*{}
        \begin{figure}[h!]
        \centering
        \subfloat[Rolling Sharpe Ratio]{\includegraphics[width=0.48\textwidth]{%s}}
        \hfill
        \subfloat[Rolling Volatility]{\includegraphics[width=0.48\textwidth]{%s}}
        \caption{Rolling Sharpe Ratio and Volatility (6 months)}
        \end{figure}

        \section*{Return Quantiles}
        \begin{figure}[h!]
        \centering
        \subfloat[3-Month]{\includegraphics[width=0.32\textwidth]{%s}}
        \hfill
        \subfloat[6-Month]{\includegraphics[width=0.32\textwidth]{%s}}
        \hfill
        \subfloat[12-Month]{\includegraphics[width=0.32\textwidth]{%s}}
        \caption{Return Quantiles Over Different Time Spreads}
        \end{figure}

        \section*{Return Analysis}
        \begin{figure}[h!]
        \centering
        \subfloat[Monthly Returns Heatmap]{\includegraphics[width=0.48\textwidth]{%s}}
        \hfill
        \subfloat[Returns Distribution]{\includegraphics[width=0.48\textwidth]{%s}}
        \caption{Monthly Returns Heatmap and Returns Distribution}
        \end{figure}

        \begin{figure}[h!]
        \centering
        \subfloat[Yearly Returns]{\includegraphics[width=0.48\textwidth]{%s}}
        \hfill
        \subfloat[Weekly Returns]{\includegraphics[width=0.48\textwidth]{%s}}
        \caption{Yearly and Weekly Returns}
        \end{figure}

        \begin{figure}[h!]
        \centering
        \includegraphics[width=0.95\textwidth]{%s}
        \caption{Drawdowns in Portfolio Value Over Time}
        \end{figure}

        \end{document}
        """ % (self.name, sharpe_ratio, max_drawdown, normalized_annual_return,
               all_total_trades, all_trades_per_year, all_average_return,
               winning_total_trades, winning_trades_per_year, winning_average_return,
               winning_best_trade, winning_max_duration, winning_min_duration,
               losing_total_trades, losing_trades_per_year, losing_average_return,
               losing_worst_trade, losing_max_duration, losing_min_duration,
               win_rate, lose_rate, profit_factor, win_loss_ratio, payoff_ratio,
               cpc_index, expectancy,
               cumulative_returns_path, portfolio_value_path,
               rolling_sharpe_path, rolling_volatility_path,
               returns_quantile_3m_path, returns_quantile_6m_path, returns_quantile_12m_path,
               monthly_returns_heatmap_path, returns_distribution_path,
               yearly_returns_path, weekly_returns_path, drawdowns_path)

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
        plt.figure(figsize=(12, 8), dpi=300)
        plt.plot(dates[:-1], portfolio_values, color='#1f77b4')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Portfolio Value (₹)', fontsize=18)
        plt.title('Portfolio Value Over Time', fontsize=21)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_cumulative_returns(self, dates, cumulative_returns, save_path):
        """
        Plot the cumulative returns over time.

        Args:
        dates (List[datetime.datetime]): Corresponding dates.
        cumulative_returns (np.array): Cumulative returns.
        save_path (str): Path to save the plot.
        """
        plt.figure(figsize=(12, 8), dpi=300)
        
        # Convert dates to matplotlib date format
        strategy_dates = mdates.date2num(dates[:-1])
        benchmark_dates = mdates.date2num(self.benchmark['Date'])
        
        # Plot strategy returns
        plt.plot_date(strategy_dates, cumulative_returns, '-', color='blue', label="Strategy return")
        
        # Plot benchmark returns
        benchmark_returns =  np.log(self.benchmark['Close']).diff().shift(-1)
        plt.plot_date(benchmark_dates, benchmark_returns.cumsum(), '-', color='#3ca12f', label="Benchmark")
        
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Cumulative Returns', fontsize=18)
        plt.title('Cumulative Returns Over Time', fontsize=21)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gcf().autofmt_xdate()  # Rotate and align the tick labels
        
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_logarithmic_returns(self, dates, log_returns, save_path):
        """
        Plot the logarithmic returns over time.

        Args:
            dates (List[datetime.datetime]): Corresponding dates.
            log_returns (np.array): Logarithmic returns.
            save_path (str): Path to save the plot.
        """
        plt.figure(figsize=(14, 10), dpi=300)
        plt.plot(dates[:-1], log_returns, label='Logarithmic Returns')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Logarithmic Returns', fontsize=18)
        plt.title('Logarithmic Returns Over Time', fontsize=21)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_returns_quantile(self, dates, returns, months, save_path):
        period_returns = [np.prod(1 + returns[i:i+months*21]) - 1 for i in range(len(returns) - months*21)]
        
        plt.figure(figsize=(12, 10), dpi=300)
        sns.boxplot(y=period_returns, color='lightblue', whis=[5, 95])
        plt.title(f'{months}-Month Returns Quantile', fontsize=21)
        plt.ylabel('Returns', fontsize=18)
        
        # Adjust y-axis limits to show outliers better
        y_min, y_max = plt.ylim()
        plt.ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_returns_distribution(self, returns, save_path):
        plt.figure(figsize=(14, 10), dpi=300)
        sns.histplot(returns, bins=50, kde=True, color='skyblue', edgecolor='black')
        plt.xlabel('Returns', fontsize=18)
        plt.ylabel('Frequency', fontsize=18)
        plt.title('Returns Distribution', fontsize=21)
        
        mean_return = np.mean(returns)
        median_return = np.median(returns)
        
        plt.axvline(mean_return, color='r', linestyle='--', label=f'Mean: {mean_return:.4f}')
        plt.axvline(median_return, color='g', linestyle='--', label=f'Median: {median_return:.4f}')
        
        # Adjust x-axis limits to show outliers better
        x_min, x_max = plt.xlim()
        plt.xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
        
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_rolling_volatility(self, dates, returns, save_path, window=126):
        df = pd.DataFrame({'date': dates[1:], 'returns': returns})
        df['rolling_vol'] = df['returns'].rolling(window).std() * np.sqrt(252)
        
        plt.figure(figsize=(14, 10), dpi=300)
        plt.plot(df['date'], df['rolling_vol'])
        plt.title(f'Rolling Volatility ({window} trading days)', fontsize=21)
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Annualized Volatility', fontsize=18)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_rolling_sharpe_ratio(self, dates, returns, save_path, window=126):
        df = pd.DataFrame({'date': dates[1:], 'returns': returns})
        df['rolling_sharpe'] = (df['returns'].rolling(window).mean() * np.sqrt(252)) / (df['returns'].rolling(window).std() * np.sqrt(252))
        
        plt.figure(figsize=(14, 10), dpi=300)
        plt.plot(df['date'], df['rolling_sharpe'])
        plt.title(f'Rolling Sharpe Ratio ({window} trading days)', fontsize=21)
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Sharpe Ratio', fontsize=18)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_monthly_returns_heatmap(self, dates, returns, save_path):
        df = pd.DataFrame({'date': dates[1:], 'returns': returns})
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

        monthly_returns = df.groupby(['year', 'month'])['returns'].apply(lambda x: (x + 1).prod() - 1).unstack()

        plt.figure(figsize=(14, 10), dpi=300)
        sns.heatmap(monthly_returns, annot=True, fmt=".2%", cmap='RdYlGn', cbar=True, 
                    center=0, linewidths=.5, square=True, annot_kws={"size": 10})
        plt.title('Monthly Returns Heatmap', fontsize=21)
        plt.xlabel('Month', fontsize=18)
        plt.ylabel('Year', fontsize=18)
        plt.yticks(rotation=0)
        plt.xticks(ticks=np.arange(0.5, 12.5), labels=[calendar.month_abbr[i] for i in range(1, 13)], rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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

    def _plot_yearly_returns(self, dates, returns, save_path):
        """
        Plot a horizontal bar graph of yearly returns.

        Args:
            dates (List[datetime.datetime]): Corresponding dates.
            returns (np.array): Array of daily returns.
            save_path (str): Path to save the plot.
        """
        df = pd.DataFrame({'date': dates[1:], 'returns': returns})
        df['year'] = df['date'].dt.year

        yearly_returns = df.groupby('year')['returns'].apply(lambda x: (x + 1).prod() - 1)

        plt.figure(figsize=(14, 10), dpi=300)
        bars = plt.barh(yearly_returns.index, yearly_returns * 100, height=0.6)
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        plt.title('Yearly Returns', fontsize=21)
        plt.xlabel('Returns (%)', fontsize=18)
        plt.ylabel('Year', fontsize=18)
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)

        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}%', ha='left', va='center', fontsize=12)

        for bar in bars:
            if bar.get_width() < 0:
                bar.set_color('red')
            else:
                bar.set_color('green')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_drawdowns(self, dates, portfolio_values, save_path):
        """
        Plot drawdowns in the portfolio value over time.

        Args:
            dates (List[datetime.datetime]): Corresponding dates.
            portfolio_values (np.array): Array of portfolio values.
            save_path (str): Path to save the plot.
        """
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        drawdown_series = pd.Series(drawdown, index=dates[:-1])
        
        # Identify drawdown periods
        in_drawdown = False
        drawdown_periods = []
        start_date = None

        for date, dd in drawdown_series.items():
            if dd < 0:
                if not in_drawdown:
                    in_drawdown = True
                    start_date = date
            else:
                if in_drawdown:
                    in_drawdown = False
                    end_date = date
                    drawdown_periods.append((start_date, end_date))
        
        # If the last period is still in drawdown, close it
        if in_drawdown:
            drawdown_periods.append((start_date, dates[-2]))

        plt.figure(figsize=(12, 8), dpi=300)
        plt.plot(dates[:-1], portfolio_values, label='Portfolio Value', color='blue')
        
        # Highlight drawdown regions
        for start_date, end_date in drawdown_periods:
            start_idx = dates.index(start_date)
            end_idx = dates.index(end_date) + 1  # Include the end date in the shading
            plt.fill_between(dates[start_idx:end_idx], portfolio_values[start_idx:end_idx], peak[start_idx:end_idx], 
                            color='red', alpha=0.3)
        
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Portfolio Value', fontsize=18)
        plt.title('Portfolio Value and Major Drawdowns Over Time', fontsize=21)
        plt.legend(fontsize=15)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


    def _plot_weekly_returns(self, dates, returns, save_path, window=24):
        """
        Plot a horizontal bar graph of composite rolling average weekly returns.

        Args:
            dates (List[datetime.datetime]): Corresponding dates.
            returns (np.array): Array of daily returns.
            save_path (str): Path to save the plot.
            window (int): Rolling window for averaging weekly returns. Defaults to 4 weeks.
        """
        df = pd.DataFrame({'date': dates[1:], 'returns': returns})
        df.set_index('date', inplace=True)
        
        # Resample the returns to weekly frequency and compute the product of returns for each week
        weekly_returns = df['returns'].resample('W').apply(lambda x: (1 + x).prod() - 1)
        
        # Compute rolling average weekly returns
        rolling_avg_weekly_returns = weekly_returns.rolling(window=window).mean()
        
        plt.figure(figsize=(14, 10), dpi=300)
        bars = plt.barh(rolling_avg_weekly_returns.index, rolling_avg_weekly_returns * 100, height=5)
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        plt.title(f'Composite Rolling Average Weekly Returns ({window}-week window)', fontsize=24)
        plt.xlabel('Returns (%)', fontsize=18)
        plt.ylabel('Week', fontsize=18)
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)

        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}%', ha='left' if width >= 0 else 'right', va='center', fontsize=12)

        for bar in bars:
            if bar.get_width() < 0:
                bar.set_color('red')
            else:
                bar.set_color('green')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _analyze_trades(self):
        closed_trades = [trade for trade in self.trade_history if trade['type'] == 'TRADE_CLOSED']
        winners = [trade for trade in closed_trades if trade['pnl'] > 0]
        losers = [trade for trade in closed_trades if trade['pnl'] <= 0]

        total_trades = len(closed_trades)
        if total_trades == 0:
            return {
                'all_trades': {},
                'winning_trades': {},
                'losing_trades': {},
                'win_rate': 0,
                'lose_rate': 0,
                'profit_factor': float('inf'),
                'win_loss_ratio': float('inf'),
                'payoff_ratio': float('inf'),
                'cpc_index': 0,
                'expectancy': 0,
            }

        total_years = (closed_trades[-1]['sell_date'] - closed_trades[0][
            'buy_date']).days / 365.25 if closed_trades else 1

        win_rate = len(winners) / total_trades
        lose_rate = len(losers) / total_trades

        # Calculate average win and loss as a percentage of the initial buy price
        average_win = np.mean([
            (trade['pnl']) for trade in winners if trade['buy_price'] != 0]) if winners else 0
        average_loss = sum(
            (trade['pnl']) for trade in losers if trade['buy_price'] != 0) / len(
            losers) if losers else 0

        best_trade = max(
            (trade['pnl'] if trade['buy_price'] != 0 else 0 for trade in winners), default=0)
        worst_trade = min(
            (trade['pnl'] if trade['buy_price'] != 0 else 0 for trade in losers), default=0)

        max_duration_winners = max(((trade['sell_date'] - trade['buy_date']).days for trade in winners if
                                    trade['sell_date'] and trade['buy_date']), default=0)
        min_duration_winners = min(((trade['sell_date'] - trade['buy_date']).days for trade in winners if
                                    trade['sell_date'] and trade['buy_date']), default=0)
        max_duration_losers = max(((trade['sell_date'] - trade['buy_date']).days for trade in losers if
                                   trade['sell_date'] and trade['buy_date']), default=0)
        min_duration_losers = min(((trade['sell_date'] - trade['buy_date']).days for trade in losers if
                                   trade['sell_date'] and trade['buy_date']), default=0)

        profit_factor = sum(trade['pnl'] for trade in winners) / abs(
            sum(trade['pnl'] for trade in losers)) if losers else float('inf')
        win_loss_ratio = win_rate / lose_rate if lose_rate > 0 else float('inf')
        payoff_ratio = average_win / abs(average_loss) if average_loss != 0 else float('inf')
        cpc_index = (win_rate * payoff_ratio) - 1
        expectancy = (win_rate * average_win) - (lose_rate * abs(average_loss))

        return {
            'all_trades': {
                'total_trades': total_trades,
                'trades_per_year': total_trades / total_years,
                'average_return_per_trade': sum(
                    (trade['pnl'] * 100 / trade['buy_price']) if trade['buy_price'] != 0 else 0 for trade in
                    closed_trades) / total_trades if total_trades > 0 else 0,
            },
            'winning_trades': {
                'total_trades': len(winners),
                'trades_per_year': len(winners) / total_years,
                'average_return_per_trade': average_win,
                'best_trade': best_trade,
                'max_duration': max_duration_winners,
                'min_duration': min_duration_winners,
            },
            'losing_trades': {
                'total_trades': len(losers),
                'trades_per_year': len(losers) / total_years,
                'average_return_per_trade': average_loss,
                'worst_trade': worst_trade,
                'max_duration': max_duration_losers,
                'min_duration': min_duration_losers,
            },
            'win_rate': win_rate,
            'lose_rate': lose_rate,
            'profit_factor': profit_factor,
            'win_loss_ratio': win_loss_ratio,
            'payoff_ratio': payoff_ratio,
            'cpc_index': cpc_index,
            'expectancy': expectancy,
        }
