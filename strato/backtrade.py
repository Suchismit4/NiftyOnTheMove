import backtrader as bt
import datetime
import logging

# Set up logging
log_filename = datetime.datetime.now().strftime("%Y-%m-%d") + "_bt.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

# Define the strategy
class MAcrossover(bt.Strategy):
    params = (
        ('fast', 10),  # fast moving average
        ('slow', 30),  # slow moving average
    )

    def __init__(self):
        self.fastma = {data: bt.indicators.SMA(self.data.close, period=self.params.fast) for data in self.datas}
        self.slowma = {data: bt.indicators.SMA(self.data.close, period=self.params.slow) for data in self.datas}
        self.crossovers = {data: bt.indicators.CrossOver(self.fastma[data], self.slowma[data]) for data in self.datas}
        self.size = 10  # Fixed size for trades
        self.order = None
        
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.data.datetime[0]
        if isinstance(dt, float):
            dt = bt.num2date(dt)
        logging.info('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            self.log('ORDER ACCEPTED/SUBMITTED', dt=order.created.dt)
            self.order = order
            return

        if order.status in [order.Expired]:
            self.log('BUY EXPIRED')

        elif order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

        # Sentinel to None: new orders allowed
        self.order = None
        
    def next(self):
        dt = self.datas[0].datetime.date(0)
        
        if self.order: 
            self.log("Pending order... skipping date")
            return 
        
        for data in self.datas:
            symbol = data._name
            price = data.close[0]
            
            if str(dt) == "2015-06-25" and symbol == "NIFTY500DUP":
                print(self.broker.getcash())
            if self.getposition(data).size <= 0:  # not in the market
                if self.crossovers[data] > 0:  # if fast crosses above slow
                    self.buy(data=data, size=self.size)
            elif self.crossovers[data] < 0:  # in the market & cross to the downside
                self.close(data=data, size=self.size)

# Create a custom data feed
class NIFTY500Data(bt.feeds.GenericCSVData):
    params = (
        ('dtformat', '%Y-%m-%d'),
        ('datetime', 0),
        ('open', 1),
        ('high', 2),
        ('low', 3),
        ('close', 4),
        ('volume', 5),
        ('enhanced_momentum', 6),
    )

# Create a cerebro entity
cerebro = bt.Cerebro()

# Add a strategy
cerebro.addstrategy(MAcrossover)

# Create a Data Feed
data = NIFTY500Data(
    dataname='NIFTY500.csv',
    fromdate=datetime.datetime(2010, 1, 1),
    todate=datetime.datetime(2020, 12, 31),
    name='NIFTY500'
)

# Create a Data Feed
data2 = NIFTY500Data(
    dataname='NIFTY500.csv',
    fromdate=datetime.datetime(2010, 1, 1),
    todate=datetime.datetime(2020, 12, 31),
    name='NIFTY500DUP'
)

# Add the Data Feed to Cerebro
cerebro.adddata(data)
cerebro.adddata(data2)

# Set our desired cash start
cerebro.broker.setcash(100000.0)

# Add a FixedSize sizer according to the stake
cerebro.addsizer(bt.sizers.FixedSize, stake=10)
cerebro.addobserver(bt.observers.Value)
cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.02)
cerebro.addanalyzer(bt.analyzers.Returns)
cerebro.addanalyzer(bt.analyzers.DrawDown)

# Print out the starting conditions
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Run over everything
results = cerebro.run()

# Print out the final result
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Plot the result
cerebro.plot()

print(f"Sharpe: {results[0].analyzers.sharperatio.get_analysis()['sharperatio']:.3f}")
print(f"Norm. Annual Return: {results[0].analyzers.returns.get_analysis()['rnorm100']:.2f}%")
print(f"Max Drawdown: {results[0].analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")