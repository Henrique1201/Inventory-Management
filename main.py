# Resetar o ambiente para o teste
obs, _ = env.reset()
total_rewards = 0

for _ in range(len(df_produto_top)):
    # O agente decide a ação baseada nas observações (features)
    action, _states = model.predict(obs, deterministic=True)
    
    # Aplicar a ação no ambiente
    obs, reward, done, truncated, info = env.step(action)
    total_rewards += reward
    
    if done:
        break

print(f"Recompensa acumulada no teste: {total_rewards}")