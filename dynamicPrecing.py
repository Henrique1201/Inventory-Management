import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DynamicPricingEnv(gym.Env):
    def __init__(self):
        super(DynamicPricingEnv, self).__init__()
        # Ação: O agente escolhe um preço entre R$ 10,00 e R$ 100,00
        self.action_space = spaces.Box(low=10, high=100, shape=(1,), dtype=np.float32)
        
        # Estado: Preço do competidor e Nível de estoque
        self.observation_space = spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32)
        
        self.state = np.array([50.0, 100.0]) # Competidor a 50, estoque a 100

    def step(self, action):
        price = action[0]
        competitor_price = self.state[0]
        
        # Simulação simples de demanda: se preço < competidor, vende mais
        demand = max(0, 100 - price + (competitor_price - price) * 2)
        vendas = min(demand, self.state[1]) # Não pode vender mais que o estoque
        
        lucro = vendas * (price - 5) # R$ 5 é o custo de produção
        
        # Atualiza estoque e muda preço do competidor aleatoriamente
        self.state[1] -= vendas
        self.state[0] += np.random.uniform(-2, 2) 
        
        terminated = self.state[1] <= 0 # Acabou o estoque
        return self.state, lucro, terminated, False, {}

    def reset(self, seed=None):
        self.state = np.array([50.0, 100.0])
        return self.state, {}