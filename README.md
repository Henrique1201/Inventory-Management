# Inventory Management RL

Este projeto utiliza Aprendizado por Reforço (Reinforcement Learning) para otimizar o gerenciamento de estoque, evitando desperdício financeiro por excesso de armazenamento (holding costs) e maximizando os lucros, prevenindo a falta de produtos (stockouts).

O modelo é treinado usando dados simulados ou estáticos do e-commerce Olist, aplicando o algoritmo **PPO (Proximal Policy Optimization)** da biblioteca `stable-baselines3`.

##  Contexto

O gerenciamento de estoque é um problema clássico de tomada de decisão sequencial. Se muito estoque for comprado, o custo de mantê-lo armazenado sobe. Caso contrário, perde-se oportunidades de venda se a demanda não puder ser atendida. 
Usar Aprendizado por Reforço permite que um agente autônomo aprenda uma "política de compras" ideal analisando características como demanda passada (lags), médias móveis e sazonalidade.

##  A Lógica por Trás

A modelagem foi feita seguindo o padrão da biblioteca `Gymnasium`:

- **Ambiente (`src/envoriment.py`)**: Define como o agente interage com o ecossistema. Ele possui limite máximo de capacidade (ex: 100 unidades) para evitar acúmulo infinito.
- **Observações (State)**: A cada dia, o modelo recebe 12 variáveis que descrevem a situação atual, incluindo demanda em dias anteriores (lag_1, lag_2), médias móveis (rolling_mean), dia da semana, se é mês de Black Friday, e o **nível de estoque atual**.
- **Ações (Action)**: Uma decisão discreta sobre **quantas unidades pedir e armazenar** para o dia vigente.
- **Recompensa (`src/reward.py`)**:
    - **Ganho (Revenue)**: Margem de lucro nas vendas realizadas.
    - **Custo (Holding Cost)**: Penalidade por manter cada unidade sobressalente no estoque num dia.
    - **Falta (Stockout Penalty)**: Penalidade alta aplicada caso a demanda do dia exceda o estoque disponível.
- **Normalização**: O uso do módulo `VecNormalize` é vital para o treinamento de redes neurais (MLP) no PPO. Ele garante que as observações e o retorno da recompensa escalem entre -1 e 1 (aproximadamente), estabilizando o gradiente.

##  Funcionalidades

- **Treinamento de Agente PPO**: Capacidade de aprender padrões de longo prazo para balancear a estocagem.
- **Simulador Integrado**: Ao final do treinamento, o `src/training.py` roda uma simulação temporal para atestar a funcionalidade e informar a Recompensa Acumulada.
- **Exportação do Modelo**: O modelo é nativamente salvo em `ppo_olist_stock_manager.zip` com seu normalizador em `vec_normalize.pkl`.

##  Como Baixar e Instalar

1. **Clone o repositório** (ou copie os arquivos para seu ambiente local).
   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd Inventory-Management
   ```

2. **Crie um ambiente virtual** (recomendado):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instale as dependências**:
   As principais bibliotecas exigidas são `stable-baselines3`, `gymnasium`, `pandas`, `numpy` e o PyTorch.
   ```bash
   pip install stable-baselines3[extra] gymnasium pandas numpy
   ```
   *(Observação: Se for focar em CPU, você pode instalar apenas a versão padrão do PyTorch).*

##  Como Executar

Para iniciar o treinamento seguido da simulação dos resultados, execute o script central de treinamento:

```bash
python3 src/training.py
```

Durante o treinamento, você verá relatórios exibindo métricas como `value_loss` e `fps`. Logo após terminar os passos definidos (`total_timesteps`), ele passará à etapa `--- Simulação de Teste ---`, exibindo a decisão real do agente dia a dia.
