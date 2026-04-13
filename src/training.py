from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envoriment import OlistInventoryEnv
import pandas as pd

df = pd.read_csv("data/dataset_final.csv")

target_product = 'aca2eb7d00ea1a7b8ebd4e68314663af'
df_top = df[(df['product_id'] == target_product) & (df['rolling_mean_7'] > 0)].copy()

print(f"Iniciando treino para o produto {target_product} com {len(df_top)} dias de dados.")

env = OlistInventoryEnv(df_top)
check_env(env) 

# Wrap the environment for normalization
vec_env = DummyVecEnv([lambda: env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=0.0003)

# Increased total timesteps to give the model time to learn the dynamics
model.learn(total_timesteps=150000)

model.save("ppo_olist_stock_manager")
vec_env.save("vec_normalize.pkl")

# Reset without VecNormalize for simulation interpretation (or we can use it but look at the underlying env)
eval_env = DummyVecEnv([lambda: OlistInventoryEnv(df_top)])
eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)
eval_env.training = False
eval_env.norm_reward = False

obs = eval_env.reset()
total_rewards = 0

print("\n--- Simulação de Teste ---")
for i in range(len(df_top)):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)
    total_rewards += eval_env.unnormalize_reward(reward)[0] if hasattr(eval_env, 'unnormalize_reward') else reward[0]
    
    if i % 50 == 0:
        print(f"Dia {i}: Ação (Pedir)={action[0]} | Recompensa Acumulada={total_rewards:.2f} | Estoque Atual={eval_env.envs[0].stock}")
    
    if done[0]:
        break

print(f"\nRecompensa total acumulada na simulação: {total_rewards:.2f}")