# Passo a passo para executar o Projeto Olist

Este roteiro apresenta a sequência correta para executar o projeto de segmentação de consumidores do Olist, desenvolvido em Python com os algoritmos **K-Means e DBSCAN**.

O projeto utiliza uma divisão dos dados em **80% para treinamento e 20% para teste**, com estado aleatório (`random_state`) definido em 42.

## 1. Abrir o projeto no VS Code

Abra a pasta principal do projeto:

```text
Projeto_Final_Kmeans_DBSCAN/
└── Projeto_Olist/
```

Abra o terminal integrado do VS Code (`Ctrl + '`) e acesse a raiz do projeto:

```powershell
cd C:\Users\User\Desktop\Projeto_Final_Kmeans_DBSCAN\Projeto_Olist
```

Confirme que o terminal está na pasta correta:

```powershell
Get-Location
```

Todos os comandos deste roteiro devem ser executados a partir da raiz do projeto.

## 2. Preparar o ambiente Python

Antes de executar os módulos, instale as dependências registradas em `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

As principais bibliotecas utilizadas são:

- Pandas;
- NumPy;
- Matplotlib;
- Seaborn;
- Scikit-learn;
- Joblib;
- FastAPI;
- Uvicorn.

Caso o projeto utilize um ambiente virtual, ative-o antes de instalar as dependências:

```powershell
.venv\Scripts\Activate.ps1
```

## 3. Disponibilizar os dados originais

Confira se os arquivos CSV do conjunto de dados Olist estão armazenados em:

```text
dados/
└── original/
```

Arquivos necessários:

```text
olist_customers_dataset.csv
olist_orders_dataset.csv
olist_order_items_dataset.csv
olist_order_payments_dataset.csv
olist_order_reviews_dataset.csv
```

Esses arquivos constituem a fonte de dados utilizada na integração e no tratamento.

## 4. Executar o pipeline de processamento e análise

Execute os módulos na sequência apresentada a seguir. Essa ordem respeita as dependências entre os arquivos e garante a geração dos dados e artefatos necessários para as etapas seguintes.

### Etapa 1 — Integração dos dados

Comando:

```powershell
python -m src.integrar_dados
```

Objetivo:

Integra as informações dos arquivos originais de clientes, pedidos, itens, pagamentos e avaliações.

Resultado esperado:

- Geração da base integrada;
- Consolidação das informações relacionadas aos consumidores;
- Disponibilização dos dados para o módulo de tratamento.

### Etapa 2 — Tratamento dos dados

Comando:

```powershell
python -m src.tratar_dados
```

Objetivo:

Realiza o tratamento da base integrada, incluindo verificações de inconsistências, duplicidades e valores ausentes, conforme as regras implementadas no módulo.

Resultado esperado:

- Geração da base tratada;
- Padronização das informações;
- Disponibilização dos dados para a preparação da modelagem.

### Etapa 3 — Preparação dos dados

Comando:

```powershell
python -m src.preparar_dados
```

Objetivo:

Prepara as variáveis utilizadas na segmentação dos consumidores.

O módulo realiza:

- Seleção das variáveis de modelagem;
- Imputação dos valores ausentes;
- Padronização das variáveis;
- Divisão dos dados em treinamento (80%) e teste (20%);
- Ajuste do pré-processador somente com os dados de treinamento.

Resultado esperado:

- Geração dos arquivos de treinamento e teste;
- Salvamento do pré-processador;
- Disponibilização dos dados para a análise exploratória e o treinamento dos modelos.

Para a base com 42.136 consumidores, a divisão esperada é:

| Conjunto      | Registros  | Percentual |
|    ---        |   ---:     |    ---:    |
| Treinamento   |  33.708    |    80%     |
|  Teste        |  8.428     |    20%     |
| **Total**     | **42.136** |  **100%**  |

### Etapa 4 — Análise exploratória

Comando:

```powershell
python -m src.analise_exploratoria
```

Objetivo:

Realiza a análise exploratória das variáveis e dos consumidores, permitindo identificar distribuições, relações, correlações e características relevantes da base.

O módulo também utiliza o pré-processador gerado na etapa anterior para produzir a comparação da padronização e a projeção PCA.

Resultado esperado:

Geração de 12 gráficos na pasta:

```text
graficos/
```

Entre eles:

- Histogramas;
- Boxplots;
- Densidades KDE;
- Matriz de correlação;
- Pairplot;
- Gráficos de dispersão;
- Análise de valores ausentes;
- Distribuição de avaliações;
- Distribuição geográfica;
- Comparação da padronização;
- Projeção PCA bidimensional.

### Etapa 5 — Treinamento dos modelos

Comando:

```powershell
python -m src.treinar_modelos
```

Objetivo:

Realiza o treinamento dos algoritmos de segmentação:

- K-Means;
- DBSCAN.

O módulo utiliza os dados preparados para treinar os modelos e selecionar os parâmetros conforme os critérios definidos no projeto.

Resultado esperado:

- Modelos treinados;
- Parâmetros selecionados;
- Artefatos necessários para avaliação e segmentação.

### Etapa 6 — Avaliação dos modelos

Comando:

```powershell
python -m src.avaliar_modelos
```

Objetivo:

Avalia o desempenho dos modelos de segmentação utilizando as métricas:

- Silhouette Score;
- Davies-Bouldin Index;
- Calinski-Harabasz Index.

Resultado esperado:

- Cálculo das métricas;
- Comparação dos resultados de K-Means e DBSCAN;
- Geração dos relatórios de avaliação, conforme implementado no módulo.

### Etapa 7 — Segmentação dos consumidores

Comando:

```powershell
python -m src.segmentar_clientes
```

Objetivo:

Aplica os modelos treinados para identificar os grupos de consumidores existentes na base.

Resultado esperado:

- Atribuição dos segmentos aos consumidores;
- Geração da base de consumidores segmentados;
- Disponibilização dos resultados para análise e utilização posterior.

A segmentação considera os 42.136 consumidores da base, conforme a implementação do módulo.

### Etapa 8 — Predição dos consumidores

Comando:

```powershell
python -m src.predizer_clientes
```

Objetivo:

Realiza as predições utilizando os modelos treinados e os dados separados para teste.

Resultado esperado:

- Predições para os 8.428 consumidores do conjunto de teste;
- Identificação dos clusters atribuídos;
- Geração dos resultados e relatórios de predição.

## 5. Executar a API

Depois de concluir o pipeline e verificar a geração dos modelos e artefatos necessários, execute a API:

```powershell
python -m uvicorn src.api:app --reload
```

Acesse a documentação interativa:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Na documentação Swagger:

1. Localize o endpoint `POST /segmentar`.
2. Clique em **Try it out**.
3. Informe os dados de um consumidor no formato solicitado pela API.
4. Clique em **Execute**.
5. Consulte a resposta e o cluster atribuído pelo K-Means.

Para encerrar a API, pressione `Ctrl + C` no terminal em que ela está sendo executada.

**Observação:** a API deve ser executada somente depois que os artefatos exigidos por sua implementação estiverem disponíveis. O endpoint `/segmentar` deve ser utilizado conforme o contrato definido no código da API.

## 6. Consultar os arquivos gerados

Após a execução do pipeline, confira as principais pastas do projeto:

| Pasta                |          Conteúdo                                               |
|  ---                 |            ---                                                  |
| `dados/original/`    | Arquivos CSV originais do Olist                                 |
| `dados/processados/` | Bases integradas, tratadas e resultados de processamento        |
| `dados/preparados/`  | Dados de treinamento e teste                                    |
| `modelos/`           | Modelos treinados e pré-processador `.joblib`                   |
| `graficos/`          | Gráficos da análise exploratória                                |
| `relatorios/`        | Métricas, estatísticas, perfis e resultados de predição         |
| `src/`               | Módulos Python responsáveis pelo processamento e pela modelagem |

Os nomes e a disponibilidade dos arquivos dependem da implementação de cada módulo.

## 7. Sequência resumida dos comandos

Para facilitar a execução, a sequência completa do pipeline é:

```powershell
python -m src.integrar_dados
python -m src.tratar_dados
python -m src.preparar_dados
python -m src.analise_exploratoria
python -m src.treinar_modelos
python -m src.avaliar_modelos
python -m src.segmentar_clientes
python -m src.predizer_clientes
```

Após a conclusão dessas etapas, inicie a API separadamente:

```powershell
python -m uvicorn src.api:app --reload
```

**Importante:** não execute o comando da API como parte de uma sequência automática no mesmo terminal, pois o servidor permanece em execução até ser interrompido.

## 8. Utilização do projeto após a primeira execução

Como o projeto já foi executado e os modelos e artefatos necessários estão disponíveis, não é necessário repetir todas as etapas a cada utilização.

Para utilizar somente a API, abra a pasta do projeto no VS Code e execute:

```powershell
python -m uvicorn src.api:app --reload
```

A execução completa do pipeline será necessária quando houver alterações nos dados originais, nas regras de tratamento, nas variáveis utilizadas ou nos procedimentos de preparação e modelagem.

Nessas situações, execute novamente as etapas na ordem indicada neste roteiro para manter a consistência entre os dados processados, o pré-processador e os modelos.

Evite executar novamente o treinamento sem necessidade, pois essa operação pode recriar os modelos e seus respectivos artefatos.

## 9. Considerações finais

Este roteiro documenta a sequência de execução do Projeto Olist, desde a disponibilização dos dados originais até a geração dos resultados de segmentação e a disponibilização da API.

A execução organizada dos módulos permite:

- Reproduzir o processamento dos dados;
- Manter a separação entre treinamento e teste;
- Evitar vazamento de informações durante a padronização;
- Gerar os artefatos necessários para a modelagem;
- Avaliar e comparar os algoritmos K-Means e DBSCAN;
- Disponibilizar os resultados por meio da API.

Este documento pode ser utilizado como guia operacional, referência para manutenção do projeto e material de apoio para a apresentação acadêmica.
