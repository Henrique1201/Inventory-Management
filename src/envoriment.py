import gymnasium as gym
from gymnasium import spaces
import numpy as np

class OlistInventoryEnv(gym.Env):
    def __init__(self, df_product, initial_stock=50):
        super(OlistInventoryEnv, self).__init__()
        
        self.df = df_product.reset_index(drop=True)
        self.max_steps = len(self.df) - 1
        self.current_step = 0
        
        self.initial_stock = initial_stock
        self.stock = initial_stock
        
        self.action_space = spaces.Discrete(101) 
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.stock = self.initial_stock
        
        observation = self._get_obs()
        return observation, {}

    def _get_obs(self):
        features = self.df.iloc[self.current_step][['lag_1', 'lag_7', 'rolling_mean_7', 'day_of_week', 'month']].values
        return np.append(features, [self.stock]).astype(np.float32)

    def step(self, action):
        self.stock += action
        
        demand = self.df.iloc[self.current_step]['quantity']
        
        units_sold = min(self.stock, demand)
        self.stock -= units_sold
        
        reward = self._calculate_reward(units_sold, self.stock, demand)
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, done, False, {}

    def _calculate_reward(self, sold, stock, demand):
        holding_cost = stock * 0.10
        profit = sold * 20.0
        penalty = max(0, demand - sold) * 40.0
        return profit - holding_cost - penalty