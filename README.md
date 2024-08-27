# NiftyOnTheMove

# Strato Backtesting Library for Equities (+ Strategy)

Welcome to the **Strato Backtesting Library**! This library is designed to help you backtest trading strategies with efficiency and precision. Whether you're a quantitative trader or a data scientist, Strato provides the tools to simulate trading strategies against historical data, analyze performance, and generate comprehensive reports.

## Table of Contents 📚

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
  - [Strato Class](#strato-class)
  - [Position Management](#position-management)
  - [Orders and Execution](#orders-and-execution)
  - [Indicators](#indicators)
  - [Strategies](#strategies)
  - [Report Generation](#report-generation)
- [Example Strategy: StocksOnTheMoveByAndrewsClenow](#example-strategy-stocksonthmovebyandrewsclenow)
- [Logging](#logging)
- [Contributing](#contributing)
- [License](#license)

## Overview

Strato is a comprehensive backtesting library that allows you to:

- **Simulate trading strategies**: Implement and test various trading strategies using historical data.
- **Analyze performance**: Generate metrics like Sharpe ratio, drawdowns, and more.
- **Generate reports**: Automatically create detailed reports, including visualizations and statistics.

## Installation

To use Strato, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/strato-backtesting.git
cd strato-backtesting
pip install -r requirements.txt
```

## Quick Start

Here’s a quick guide to running your first backtest:

1. Load your data: Prepare your historical data in a CSV format.
2. Define your strategy: Implement your custom strategy or use a pre-defined one.
3. Run the backtest: Initialize the `Strato` class, add indicators, and execute the backtest.

Example
```py
from strato import Strato
from strategy import StocksOnTheMoveByAndrewsClenow
from indicators import Momentum, ExponentialMovingAverage, AverageTrueRange, Volatility, Close

# Load data
data, symbol_to_index, feature_to_index, date_to_index, start_date, end_date, features = load_and_transform_csv('./data/NIFTY500_2010_2020_HISTORICAL.csv')

# Create strategy instance
long_term_momentum = StocksOnTheMoveByAndrewsClenow(
    constituents=constituents,  
    symbol_to_index=symbol_to_index,
    date_to_index=date_to_index,
    lookback=120,
    sma_short=100,
    sma_long=200,
    volatility=25,
    portfolio_at_risk=0.001,
    min_momentum=0.,
    max_stocks=15
)

# Initialize Strato
strato = Strato(
    data=data,
    symbol_to_index=symbol_to_index,
    feature_to_index=feature_to_index,
    date_to_index=date_to_index,
    name="NiftyOnTheMove",
    starting_cash=1000000.0,
    strategies=[long_term_momentum],
    benchmark=bchk,
    generate_report=True
)

# Add indicators
strato.add_indicator('Momentum_120', Momentum(120))
strato.add_indicator('SMA_100', ExponentialMovingAverage(100))
strato.add_indicator('SMA_200', ExponentialMovingAverage(200))
strato.add_indicator('ATR_25', AverageTrueRange(25))
strato.add_indicator('Volatility_25', Volatility(25))
strato.add_indicator('Close', Close())

# Run backtest
results = strato.run_backtest()

print(f'Final Portfolio Value: ₹{results[-1]:.2f}')
```

## Core Concepts

## Strato Class
The `Strato` class is the heart of the backtesting library. It orchestrates the simulation of trading strategies, manages positions, handles orders, and tracks performance metrics.

**Initialization**
```py
def __init__(self, data: np.ndarray, 
             symbol_to_index: Dict[str, int], 
             feature_to_index: Dict[str, int], 
             date_to_index: Dict[datetime.datetime, int], 
             starting_cash: float, 
             strategies: List[Strategy], 
             benchmark: pd.DataFrame = None, 
             generate_report: bool = False,
             name: str = None):
```

- `data`: A 3D numpy array containing historical market data.
- `symbol_to_index`: Mapping of symbols to indices in the data array.
- `feature_to_index`: Mapping of features to indices in the data array.
- `date_to_index`: Mapping of dates to indices in the data array.
- `starting_cash`: The initial cash balance.
- `strategies`: A list of strategies to be backtested.
- `benchmark`: An optional DataFrame for benchmark comparison.
- `generate_report`: If set to True, generates a detailed report post-backtest.
- `name`: An optional name for the backtest run.

## Position Management

Positions represent the assets held during the backtest. The `Position` class manages the quantity, price, and other attributes associated with a trading position.

Key Methods

- buy(quantity, price, date): Buy more of the asset, increasing the position.
- sell(quantity, price, date): Sell part or all of the position using FIFO.
- calculate_value(current_price): Calculate the current market value of the position.
- calculate_pnl(current_price): Calculate the unrealized profit and loss.

Example
```py
position = Position('AAPL', 10, 150.0, datetime.datetime.now())
position.buy(5, 155.0, datetime.datetime.now())
position.sell(8, 160.0, datetime.datetime.now())
current_value = position.calculate_value(162.0)
unrealized_pnl = position.calculate_pnl(162.0)
```

### Important Details:

- **FIFO Execution**: When selling, the library uses a First In, First Out (FIFO) method. This means that the oldest shares are sold first, which affects the realized profit/loss calculation.
- **Position Updates**: Positions are updated daily based on new data and existing orders. The `bars_since_entry` attribute tracks how long a position has been held.
- **Handling Missing Data**: If data for a particular symbol is missing, the last valid price is used for calculations. If a valid price cannot be found, the position may be sold at the last observed price to avoid holding a position with unreliable data.

## Orders and Execution
Orders are the actionable instructions generated by strategies. They can either be buy or sell orders.

### Order Placement vs. Execution

**Order Placement**: Orders are generated by strategies based on the previous day's data (e.g., signals generated after the market close).

**Order Execution**: Orders are executed at the market's next open price. This reflects a realistic scenario where decisions made at the close of one trading day are executed at the open of the next.

### Order Execution

Orders are executed based on market data. The Order class handles the mechanics of executing trades, adjusting positions, and logging the results.

Example:
```py
order = Order(Strategy.BUY, date_idx, quantity, position, symbol_idx)
value, qty, execution_price, order_type, realized_pnl, lots = order.execute(daily_data, feature_to_index, order_date)
```

- Execution Price: By default, the execution price is the opening price of the next trading day. This is essential to note for strategies that rely on precise entry and exit points.
- Realized PnL: Upon order execution, the realized profit/loss is calculated and logged. This value is critical for performance evaluation.

## Indicators
Indicators
Indicators are the building blocks for strategies. The library includes several pre-defined indicators like Moving Averages, Volatility, and Momentum.

**Custom Indicator Example: Exponential Moving Average**
```py
class ExponentialMovingAverage(Indicator):
    def __init__(self, window: int):
        self.window = window
        self.alpha = 2 / (window + 1)  # Smoothing factor

    def init(self, data, feature_to_index):
        # Initialize the EMA with historical data
        ...

    def step(self, current_value, new_data, carry):
        # Update EMA with new data
        ...
```

## Strategies
Strategies define the trading logic. They decide when to buy or sell based on indicator values and other market conditions.

**Example Strategy: StocksOnTheMoveByAndrewsClenow**

This strategy ranks stocks based on momentum and executes trades accordingly.
```py 
class StocksOnTheMoveByAndrewsClenow(Strategy):
    def __init__(self, constituents, symbol_to_index, date_to_index, ...):
        super().__init__()
        ...

    def step(self, date_idx, indicator_calculator, positions, symbol_to_index, benchmark=None):
        # Generate trading signals and execute orders
        ...
```

# **Stocks On The Move** _by_ Andrews Clenow

The StocksOnTheMoveByAndrewsClenow strategy is an implementation based on Andreas Clenow’s momentum strategy. It ranks stocks by momentum and uses trailing stops to exit positions.

### Strategy Logic
- **Ranking**: Stocks are ranked based on their momentum.
- **Rebalancing**: Positions are rebalanced regularly to maintain a portfolio of the top-ranked stocks.
- **Re-construction**: Assets are sold/bought to maintain satisfactory holding conditions.

### Important Details:

- **Handling Gaps and Volatility**: The strategy takes into account potential price gaps between the previous close and the next open. This is particularly relevant for stop-loss orders and rebalancing decisions.
- **Portfolio Constraints**: The strategy enforces constraints like the maximum number of stocks in the portfolio and the minimum cash balance required for new positions.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

### License
#### This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.