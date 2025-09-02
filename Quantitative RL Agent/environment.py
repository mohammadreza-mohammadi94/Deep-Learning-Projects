# environment.py
"""
Contains the MultiStockEnv class for the stock trading reinforcement learning environment.
"""
import numpy as np
import itertools
from typing import List, Tuple, Dict, Any

class MultiStockEnv:
    """
    A multi-stock trading environment.
    
    State: (#shares_1, ..., #shares_N, price_1, ..., price_N, cash)
    Action: 3^N possibilities (buy, sell, hold for each stock)
    """
    def __init__(self, data: np.ndarray, initial_investment: float = 20000.0):
        self.stock_price_history = data
        self.n_step, self.n_stock = self.stock_price_history.shape
        self.initial_investment = initial_investment
        
        # Action space setup
        self.action_space = np.arange(3**self.n_stock)
        self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))
        
        # State space dimension
        self.state_dim = self.n_stock * 2 + 1
        
        self.reset()

    def reset(self) -> np.ndarray:
        """Resets the environment to the initial state."""
        self.cur_step = 0
        self.stock_owned = np.zeros(self.n_stock)
        self.stock_price = self.stock_price_history[self.cur_step]
        self.cash_in_hand = self.initial_investment
        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Perform one step in the environment."""
        assert action in self.action_space

        prev_val = self._get_val()
        self._trade(action)
        
        self.cur_step += 1
        if self.cur_step >= self.n_step:
            # Prevent index out of bounds
            self.cur_step = self.n_step - 1

        self.stock_price = self.stock_price_history[self.cur_step]
        cur_val = self._get_val()
        
        reward = cur_val - prev_val
        done = self.cur_step == self.n_step - 1
        info = {'cur_val': cur_val}

        return self._get_obs(), reward, done, info

    def _get_obs(self) -> np.ndarray:
        """Constructs the observation state."""
        obs = np.empty(self.state_dim)
        obs[:self.n_stock] = self.stock_owned
        obs[self.n_stock:2*self.n_stock] = self.stock_price
        obs[-1] = self.cash_in_hand
        return obs

    def _get_val(self) -> float:
        """Calculates the current portfolio value."""
        return self.stock_owned.dot(self.stock_price) + self.cash_in_hand

    def _trade(self, action: int):
        """
        Executes a trade based on the given action.
        Action mapping: 0=sell, 1=hold, 2=buy.
        """
        action_vec = self.action_list[action]
        sell_indices = [i for i, a in enumerate(action_vec) if a == 0]
        buy_indices = [i for i, a in enumerate(action_vec) if a == 2]

        # Sell all shares of selected stocks
        if sell_indices:
            for i in sell_indices:
                self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
                self.stock_owned[i] = 0

        # Buy one share at a time for selected stocks until cash runs out
        if buy_indices:
            can_buy = True
            while can_buy:
                for i in buy_indices:
                    if self.cash_in_hand > self.stock_price[i]:
                        self.stock_owned[i] += 1
                        self.cash_in_hand -= self.stock_price[i]
                    else:
                        # Not enough cash to buy even one more share of the cheapest stock in the buy list
                        can_buy = False