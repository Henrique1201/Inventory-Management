import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import os

# --- 1. Definição do Ambiente ---
class OlistInventoryEnv(gym.Env):
    def __init__(self, df_product, initial_stock=20):
        super(OlistInventoryEnv, self).__init__()
        # Limpar o dataset e resetar o índice
        self.df = df_product.reset_index(drop=True)
        self.max_steps = len(self.df) - 1
        self.current_step = 0
        self.stock = initial_stock
        
        # Espaço de Observação: 11 colunas do CSV + 1 coluna do estoque atual = 12
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        
        # Espaço de Ação: O agente pode pedir de 0 a 50 unidades de reposição
        self.action_space = spaces.Discrete(51)

    def _get_obs(self):
        # Colunas extraídas do seu Feature Engineering
        cols = ['lag_1', 'lag_2', 'lag_3', 'lag_7', 'rolling_mean_7', 
                'rolling_std_7', 'diff_1', 'day_of_week', 'month', 
                'is_weekend', 'is_black_friday_month']
        features = self.df.iloc[self.current_step][cols].values
        # Adiciona o estoque atual ao vetor que o agente "enxerga"
        obs = np.append(features, [self.stock]).astype(np.float32)
        return obs

    def step(self, action):
        # 1. Reposição do estoque (Ação do agente)
        self.stock += action
        
        # 2. Demanda real do dia (do dataset)
        demand = self.df.iloc[self.current_step]['quantity']
        
        # 3. Cálculo de Vendas e Estoque Restante
        units_sold = min(self.stock, demand)
        self.stock -= units_sold
        
        # 4. Cálculo da Recompensa (Lucro vs Prejuízo)
        # Margem: R$ 20 | Custo Estoque: R$ 0.10 | Penalidade Falta: R$ 40
        profit = units_sold * 20.0
        holding_cost = self.stock * 0.10
        stockout_penalty = max(0, demand - units_sold) * 40.0
        
        reward = profit - holding_cost - stockout_penalty
        
        # Avançar no tempo
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.stock = 20
        return self._get_obs(), {}

# --- 2. Preparação dos Dados e Treinamento ---

