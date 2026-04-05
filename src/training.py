from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from envoriment import OlistInventoryEnv
import pandas as pd

# Carregar o dataset (ajuste o caminho se necessário)
df = pd.read_csv("data/dataset_final.csv")

# Focar no produto com mais vendas para o agente aprender mais rápido
target_product = 'aca2eb7d00ea1a7b8ebd4e68314663af'
df_top = df[(df['product_id'] == target_product) & (df['rolling_mean_7'] > 0)].copy()

print(f"Iniciando treino para o produto {target_product} com {len(df_top)} dias de dados.")

# Inicializar ambiente
env = OlistInventoryEnv(df_top)
check_env(env) # Valida se o ambiente está correto

# Configurar o modelo PPO
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

# Treinar por 20.000 passos (o agente verá o histórico várias vezes)
model.learn(total_timesteps=20000)

# Salvar o modelo treinado
model.save("ppo_olist_stock_manager")

# --- 3. Teste de Performance (Simulação) ---
obs, _ = env.reset()
total_rewards = 0

print("\n--- Simulação de Teste ---")
for i in range(len(df_top)):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_rewards += reward
    
    if i % 50 == 0:
        print(f"Dia {i}: Ação (Pedir)={action} | Recompensa Acumulada={total_rewards:.2f}")
    
    if done:
        break

print(f"\nRecompensa total acumulada na simulação: {total_rewards:.2f}")