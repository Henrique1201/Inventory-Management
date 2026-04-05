from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# 1. Instanciar o ambiente com os dados de um produto específico (Classe A)
# Supondo que 'df_produto_top' é o DataFrame que preparamos com as features
env = OlistInventoryEnv(df_produto_top)

# 2. Verificar se o ambiente segue os padrões do Gymnasium (Boa prática!)
check_env(env)

# 3. Criar o Modelo PPO
# 'MlpPolicy' significa que usaremos uma rede neural simples (Multi-layer Perceptron)
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

# 4. Treinar o agente! 
# Ele vai "jogar" o cenário de vendas da Olist 10.000 vezes para aprender
model.learn(total_timesteps=10000)

# 5. Salvar o cérebro do seu agente
model.save("ppo_olist_stock_manager")