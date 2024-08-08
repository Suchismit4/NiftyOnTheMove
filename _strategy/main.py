import pandas as pd
import datetime
import logging
import numpy as np
from typing import Dict, Optional
from multiprocessing import Pool
import backtrader as bt
from tqdm import tqdm
from scipy.stats import linregress

# Set up logging
log_filename = datetime.datetime.now().strftime("%Y-%m-%d") + "_bt.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

class Momentum(bt.Indicator):
    lines = ('trend',)
    params = (('period', 90),)
    
    def __init__(self):
        self.addminperiod(self.params.period)
    
    def next(self):
        returns = np.log(self.data.get(size=self.p.period))
        x = np.arange(len(returns))
        slope, _, rvalue, _, _ = linregress(x, returns)
        annualized = (1 + slope) ** 252
        self.lines.trend[0] = annualized * (rvalue ** 2)

class NiftyOnTheMove(bt.Strategy):
    params = (
        ('fast', 100),  # fast moving average
        ('slow', 200),  # slow moving average
    )
    
    def __init__(self):
        self.i = 0
        self.inds = {}
        self.nifty = self.datas[0]
        self.stocks = self.datas[1:]
        
        self.nifty_sma200 = bt.indicators.SimpleMovingAverage(self.nifty.close,
                                                              period=200)
        
        # Load NIFTY500 composition data
        self.index_data = pd.read_csv('NIFTY500_2010_2020.csv', parse_dates=['Event Date'])
        self.index_data.set_index('Event Date', inplace=True)
        
        for d in self.stocks:
            self.inds[d] = {}
            self.inds[d]["momentum"] = Momentum(d.close, period=90)
            self.inds[d]["sma100"] = bt.indicators.SimpleMovingAverage(d.close, period=100)
            self.inds[d]["atr20"] = bt.indicators.ATR(d, period=20)
        
        print("Ready to backtest...")
        
    def prenext(self):
        # call next() even when data is not available for all tickers
        self.next()
        
    def next(self):
        if self.i % 5 == 0:
            print('Portfolio rebalanced')
            self.rebalance_portfolio()
        if self.i % 10 == 0:
            self.rebalance_positions()
            print('Positions rebalanced')
        self.i += 1
        
    def get_composition(self, date):
        date_str = date.strftime('%Y-%m-%d')
        if date_str in self.index_data.index:
            symbols = self.index_data.loc[date_str, 'ticker'].unique()
            return symbols
        return []
    
    def rebalance_portfolio(self):
        current_date = self.data.datetime.date(0)
        allowed_symbols = self.get_composition(current_date)
        
        # only look at data that we can have indicators for and that are in the index composition
        self.rankings = [d for d in self.stocks if d._name in allowed_symbols and len(d) > 100]
        # Filter based on that any value in the d must not be nan or zero for closing price 
        self.rankings.sort(key=lambda d: self.inds[d]["momentum"][0], reverse=True)
        num_stocks = len(self.rankings)

        # sell stocks based on criteria
        for i, d in enumerate(self.rankings):
            if self.getposition(d).size:
                if i > num_stocks * 0.2 or d.close[0] < self.inds[d]["sma100"]:
                    self.close(d)
                    
        if self.nifty.close[0] < self.nifty_sma200[0]:
            return
        
        # buy stocks with remaining cash
        for i, d in enumerate(self.rankings[:int(num_stocks * 0.2)]):
            cash = self.broker.get_cash()
            value = self.broker.get_value()
            if cash <= 0:
                break
            if not self.getposition(d).size:
                size = value * 0.001 / self.inds[d]["atr20"]
                self.buy(d, size=size)
                
    def rebalance_positions(self):
        current_date = self.data.datetime.date(0)
        allowed_symbols = self.get_composition(current_date)
        
        num_stocks = len(self.rankings)
        
        if self.nifty.close[0] < self.nifty_sma200[0]:
            return

        # rebalance all stocks
        for i, d in enumerate(self.rankings[:int(num_stocks * 0.2)]):
            if d._name not in allowed_symbols:
                continue
            cash = self.broker.get_cash()
            value = self.broker.get_value()
            if cash <= 0:
                break
            size = value * 0.001 / self.inds[d]["atr20"]
            self.order_target_size(d, size)

cerebro = bt.Cerebro(stdstats=False)
cerebro.broker.set_coc(True)

nifty = pd.read_csv('NIFTY500.csv', parse_dates=True, index_col=0)

cerebro.adddata(bt.feeds.PandasData(dataname=nifty, plot=False))

def load_data(filename):
    symbol_path = os.path.join("./data/", filename)
    df = pd.read_csv(symbol_path, parse_dates=True, index_col=0)
    cerebro.adddata(bt.feeds.PandasData(dataname=df, plot=False))

import os
files = [f for f in os.listdir("./data/") if f.endswith('.csv')]
for f in tqdm(files):
    load_data(filename=f)
    
cerebro.addobserver(bt.observers.Value)
cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)
cerebro.addanalyzer(bt.analyzers.Returns)
cerebro.addanalyzer(bt.analyzers.DrawDown)
cerebro.addstrategy(NiftyOnTheMove)
results = cerebro.run()
cerebro.plot()
print(f"Sharpe: {results[0].analyzers.sharperatio.get_analysis()['sharperatio']:.3f}")
print(f"Norm. Annual Return: {results[0].analyzers.returns.get_analysis()['rnorm100']:.2f}%")
print(f"Max Drawdown: {results[0].analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")