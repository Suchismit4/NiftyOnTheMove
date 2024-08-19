import numpy as np

class Broker():

    def __init__(self):
        self.cash = -np.inf
        self.portfolio_value = -np.inf
        pass

    def set(self, cash, portfolio_value):
        self.cash = cash
        self.portfolio_value = portfolio_value

    def get_disposable_cash(self):
        return self.cash

    def get_portfolio_value(self):
        return self.portfolio_value