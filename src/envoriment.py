import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os
from reward import calculate_reward

class OlistInventoryEnv(gym.Env):
    def __init__(self, df_product, initial_stock=20, max_capacity=100):
        super(OlistInventoryEnv, self).__init__()
        self.df = df_product.reset_index(drop=True)
        self.max_steps = len(self.df) - 1
        self.current_step = 0
        self.stock = initial_stock
        self.max_capacity = max_capacity
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(51)

    def _get_obs(self):
        cols = ['lag_1', 'lag_2', 'lag_3', 'lag_7', 'rolling_mean_7', 
                'rolling_std_7', 'diff_1', 'day_of_week', 'month', 
                'is_weekend', 'is_black_friday_month']
        features = self.df.iloc[self.current_step][cols].values
        obs = np.append(features, [self.stock]).astype(np.float32)
        return obs

    def step(self, action):
        # Prevent stocking beyond max capacity
        ordered_units = min(action, self.max_capacity - self.stock)
        self.stock += ordered_units
        
        demand = self.df.iloc[self.current_step]['quantity']
        
        units_sold = min(self.stock, demand)
        self.stock -= units_sold
        
        # Using the reward function from reward.py
        # Passed a unit_price of 66.66 to keep margin_per_unit ~ 20.0
        reward = calculate_reward(units_sold, self.stock, demand, unit_price=66.66)
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.stock = 20
        return self._get_obs(), {}

