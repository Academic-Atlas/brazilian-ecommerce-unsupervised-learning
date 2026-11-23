# Projeto Olist - Segmentação de Clientes com Aprendizado Não Supervisionado

## 1. Objetivo

Este projeto realiza a integração, a preparação e a análise de dados de pedidos do **Brazilian E-Commerce Public Dataset (Olist)**, com o objetivo de segmentar consumidores utilizando dois algoritmos de aprendizado não supervisionado: **K-Means** e **DBSCAN**.

A unidade de análise é o consumidor identificado por `customer_unique_id`. A análise considera pedidos entregues no período de **1º de janeiro a 31 de dezembro de 2017**.

## 2. Estrutura do projeto

```text
Projeto_Olist/
├── dados/
│   ├── original/       # Arquivos CSV originais do Olist
│   ├── preparados/     # Matrizes de treino/teste e IDs do teste
│   └── processados/    # Bases integradas, tratadas e segmentadas
├── modelos/            # Modelos e pré-processador serializados
├── relatorios/         # Métricas, estatísticas e resultados em CSV/TXT
├── graficos/           # Pasta mantida; não foram gerados gráficos nesta versão
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── integrar_dados.py
│   ├── tratar_dados.py
│   ├── analise_exploratoria.py
│   ├── preparar_dados.py
│   ├── treinar_modelos.py
│   ├── avaliar_modelos.py
│   ├── segmentar_clientes.py
│   ├── predizer_clientes.py
│   └── api.py
├── requirements.txt
└── README.md
```

## 3. Requisitos

- Python 3.13 (versão utilizada na execução deste projeto).
- Pacotes listados em `requirements.txt`.
- Arquivos originais do Olist na pasta `dados/original/`.

Instale as dependências na raiz do projeto:

```powershell
python -m pip install -r requirements.txt
```

Para executar a API, FastAPI e Uvicorn também precisam estar instalados. Ambos estão declarados em `requirements.txt`.

## 4. Arquivos de entrada

A configuração do projeto espera os seguintes arquivos em `dados/original/`:

- `olist_customers_dataset.csv`
- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_order_payments_dataset.csv`
- `olist_order_reviews_dataset.csv`

Confira se os nomes dos arquivos estão exatamente iguais aos indicados acima.

## 5. Como executar o pipeline

Execute os comandos abaixo **a partir da pasta raiz `Projeto_Olist`**, no terminal do VS Code ou no PowerShell. A forma `python -m src.nome_do_modulo` permite que os imports do pacote `src` sejam resolvidos corretamente.

### Etapa 1 — Integração dos dados

```powershell
python -m src.integrar_dados
```

Resultado validado: **43.428 pedidos entregues**, **42.136 consumidores** e base integrada com dimensões `(42136, 14)`.

### Etapa 2 — Tratamento dos dados

```powershell
python -m src.tratar_dados
```

Resultado validado: 42.136 registros antes e depois do tratamento, sem duplicidades. Foram identificados 312 valores ausentes em `avaliacao_media`; esses valores são tratados pelo imputador mediano ajustado na etapa de preparação.

### Etapa 3 — Análise exploratória

```powershell
python -m src.analise_exploratoria
```

Gera estatísticas descritivas, análise de valores ausentes, correlações e distribuição por estado nos arquivos da pasta `relatorios/`.

### Etapa 4 — Preparação e divisão dos dados

```powershell
python -m src.preparar_dados
```

A divisão validada foi de 80% para treinamento e 20% para teste:

- **Treinamento:** 33.708 consumidores.
- **Teste:** 8.428 consumidores.

O imputador e o escalonador são ajustados com os dados de treinamento. O conjunto de teste utiliza o mesmo pré-processador, sem novo ajuste. Os arquivos `X_treino.csv` e `X_teste.csv` já contêm os dados transformados.

### Etapa 5 — Treinamento dos modelos

```powershell
python -m src.treinar_modelos
```

O K-Means avaliou valores de K entre 2 e 8 e selecionou **K = 2**. O DBSCAN avaliou diferentes valores de `eps` e selecionou **`eps = 2.0`**, com `min_samples = 10`.

### Etapa 6 — Avaliação

```powershell
python -m src.avaliar_modelos
```

A avaliação validada no conjunto de teste foi:

| Algoritmo | Registros | Clusters | Ruído | Silhouette | Davies-Bouldin | Calinski-Harabasz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| K-Means | 8.428 | 2 | 0,00% | 0,597533 | 1,207780 | 1.787,57 |
| DBSCAN (atribuição aproximada) | 8.428 | 4 | 1,8272% | 0,550459 | 0,580806 | 836,88 |

O K-Means apresentou valores superiores de Silhouette e Calinski-Harabasz. O DBSCAN apresentou menor Davies-Bouldin. Portanto, os indicadores não apontam superioridade absoluta de um algoritmo em todas as métricas. O K-Means foi adotado como modelo principal para a API pela separação observada e pela simplicidade de atribuição de novos consumidores; o DBSCAN permanece como abordagem complementar para análise de densidade e identificação de ruído.

### Etapa 7 — Segmentação da base completa

```powershell
python -m src.segmentar_clientes
```

O módulo aplica os artefatos treinados à base completa de 42.136 consumidores e gera os arquivos de segmentação e os perfis dos grupos em `dados/processados/`.

#### K-Means — base completa

| Cluster | Consumidores | Participação aproximada | Valor total médio |
| ---: | ---: | ---: | ---: |
| 0 | 40.074 | 95,11% | R$ 127,25 |
| 1 | 2.062 | 4,89% | R$ 883,65 |

O cluster 1 representa um grupo menor com valor total médio de compras consideravelmente maior. Essa interpretação é descritiva e deve ser considerada em conjunto com as demais variáveis do perfil.

#### DBSCAN — base completa

| Rótulo | Consumidores | Participação aproximada | Interpretação descritiva |
| ---: | ---: | ---: | --- |
| -1 | 818 | 1,94% | Observações classificadas como ruído; não constituem um cluster regular |
| 0 | 40.041 | 95,03% | Grupo predominante |
| 1 | 967 | 2,29% | Consumidores com frequência média de 2 pedidos |
| 2 | 44 | 0,10% | Pequeno grupo com frequência média de 3 pedidos |
| 3 | 266 | 0,63% | Grupo sem avaliação registrada |

### Etapa 8 — Predição do conjunto de teste

```powershell
python -m src.predizer_clientes
```

Resultado validado para 8.428 consumidores:

- **K-Means:** cluster 0 = 8.033 consumidores; cluster 1 = 395 consumidores.
- **DBSCAN:** ruído = 154; cluster 0 = 8.031; cluster 1 = 185; cluster 2 = 11; cluster 3 = 47.
- **Taxa de ruído DBSCAN no conjunto de teste:** 1,83%.

O módulo gera `predicoes_clientes.csv` e `resumo_predicoes.csv` em `relatorios/`.

## 6. Artefatos principais

A pasta `modelos/` deve conter os artefatos utilizados nesta execução:

- `preprocessador.joblib`
- `kmeans.joblib`
- `dbscan.joblib`
- `dbscan_core.joblib`

Arquivos antigos de experimentos anteriores, caso ainda estejam na pasta, não devem ser confundidos com os artefatos utilizados pelo pipeline atual. A configuração vigente (`src/config.py`) define os caminhos oficiais.

## 7. Executar a API

A API utiliza o **K-Means** como modelo principal. O DBSCAN é mantido na análise do projeto e na rotina de predição do conjunto de teste; sua atribuição para novos dados é aproximada porque o DBSCAN não oferece um método nativo `predict`.

Na raiz do projeto, execute:

```powershell
python -m uvicorn src.api:app --reload
```

Quando o servidor iniciar, acesse a documentação interativa:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

O endpoint `POST /segmentar` recebe as dez variáveis de entrada utilizadas pelo modelo. Exemplo de corpo JSON:

```json
{
  "recencia": 100,
  "frequencia": 1,
  "valor_total": 150.00,
  "ticket_medio": 150.00,
  "total_itens": 1,
  "valor_produtos_total": 125.00,
  "frete_medio": 25.00,
  "avaliacao_media": 4.5,
  "possui_avaliacao": 1,
  "parcelas_media": 3
}
```

Resposta de teste validada:

```json
{
  "algoritmo": "K-Means",
  "grupo_cliente": 0,
  "observacao": "O número do grupo é um identificador técnico. Consulte o perfil dos clusters para sua interpretação comercial."
}
```

O número do cluster é um rótulo técnico, não uma classificação comercial autossuficiente. Consulte os arquivos de perfil dos clusters antes de atribuir nomes ou ações comerciais aos grupos.

Para encerrar o servidor, volte ao terminal em que o Uvicorn está em execução e pressione `Ctrl + C`.

## 8. Decisões metodológicas e limitações

- Foram considerados somente pedidos com status `delivered` no ano de 2017.
- A unidade de análise é `customer_unique_id`, com uma linha por consumidor.
- A data de referência para o cálculo de recência é `2018-01-01`.
- A divisão entre treinamento e teste foi de 80%/20%, com `random_state = 42`.
- A imputação pela mediana e a padronização foram ajustadas no conjunto de treinamento e reutilizadas no teste.
- As variáveis `valor_total` e `valor_produtos_total` apresentaram correlação elevada (aproximadamente 0,997), assim como `valor_total` e `ticket_medio` (aproximadamente 0,985). Essa redundância deve ser considerada na interpretação dos resultados.
- O DBSCAN foi treinado em uma amostra limitada a até 15.000 observações, em função dos recursos computacionais disponíveis. A atribuição a novos registros é aproximada, pela proximidade do ponto central mais próximo dentro de `eps`; registros fora desse limite são tratados como ruído (`-1`).
- As métricas do DBSCAN são calculadas desconsiderando os pontos classificados como ruído. A comparação com o K-Means deve considerar essa diferença.
- A segmentação identifica padrões estatísticos da base analisada; não demonstra causalidade nem garante comportamento futuro dos consumidores.

## 9. Principais arquivos de saída

### Arquivos em `dados/processados/`

- `clientes_2017_integrado.csv`
- `clientes_2017_tratado.csv`
- `clientes_segmentados_kmeans.csv`
- `clientes_segmentados_dbscan.csv`
- `perfil_clusters_kmeans.csv`
- `perfil_clusters_dbscan.csv`

### Arquivos em `dados/preparados/`

- `X_treino.csv`
- `X_teste.csv`
- `ids_teste.csv`

### Arquivos em `relatorios/`

- `metricas_modelos.csv`
- `experimentos_kmeans.csv`
- `experimentos_dbscan.csv`
- `estatisticas_descritivas.csv`
- `valores_ausentes.csv`
- `correlacao_variaveis.csv`
- `clientes_por_estado.csv`
- `resumo_tratamento.txt`
- `predicoes_clientes.csv`
- `resumo_predicoes.csv`

---

**Status da execução:** integração, tratamento, análise exploratória, preparação, treinamento, avaliação, segmentação da base completa, predição do conjunto de teste e teste básico do endpoint FastAPI foram executados com sucesso nesta versão do projeto.
