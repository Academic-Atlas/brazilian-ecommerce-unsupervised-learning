###############################################################################
# PROJETO OLIST
# ANÁLISE EXPLORATÓRIA DE DADOS (EDA)
#
# Objetivo:
# Investigar distribuição, qualidade, relacionamentos e comportamento das
# variáveis utilizadas na segmentação de clientes.
#
# Metodologia:
# CRISP-DM - Fase 2: Compreensão dos Dados
#
# Modelos posteriores: K-Means e DBSCAN
# Divisão experimental: 80% treinamento / 20% teste
#
# Este módulo é exclusivamente exploratório. Não treina modelos de clustering.
###############################################################################

# ============================================================================
# 1. IMPORTAÇÕES
# ============================================================================

import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from src.config import (
    CLEAN_FILE,
    FEATURES,
    ROOT_DIR,
    ID_COLUMN,
    INTEGRATED_FILE,
    PREPROCESSOR_FILE,
    RANDOM_STATE,
    TEST_SIZE,
    create_directories,
)


# ============================================================================
# 2. CONFIGURAÇÕES
# ============================================================================

GRAFICOS_DIR = ROOT_DIR / "graficos"

TAMANHO_AMOSTRA = 1500
LIMIAR_VARIANCIA = 1e-8

sns.set_theme(
    style="whitegrid",
    palette="deep",
)

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 300,
})


# ============================================================================
# 3. PREPARAÇÃO DE DIRETÓRIOS E UTILITÁRIOS
# ============================================================================

def preparar_diretorios():
    """Cria os diretórios necessários ao projeto e aos gráficos."""
    create_directories()
    GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)


def salvar_grafico(figura, nome_arquivo):
    """Ajusta, salva e fecha uma figura."""
    caminho = GRAFICOS_DIR / nome_arquivo

    figura.tight_layout()
    figura.savefig(
        caminho,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figura)

    print(f"Gráfico gerado: {nome_arquivo}")


def preparar_dados(dados):
    """Seleciona e converte as variáveis de modelagem para formato numérico."""
    if dados is None:
        raise ValueError("A base de dados não foi fornecida.")

    colunas_ausentes = [
        coluna for coluna in FEATURES
        if coluna not in dados.columns
    ]

    if colunas_ausentes:
        raise ValueError(
            f"Variáveis não encontradas na base: {colunas_ausentes}"
        )

    dados_numericos = dados.loc[:, FEATURES].copy()

    for coluna in FEATURES:
        dados_numericos[coluna] = pd.to_numeric(
            dados_numericos[coluna],
            errors="coerce",
        )

    return dados_numericos.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def obter_amostra(dados, tamanho=TAMANHO_AMOSTRA):
    """Retorna uma amostra reprodutível quando a base excede o limite."""
    if len(dados) <= tamanho:
        return dados.copy()

    return dados.sample(
        n=tamanho,
        random_state=RANDOM_STATE,
    )


def carregar_preprocessador():
    """Carrega o pré-processador ajustado no conjunto de treinamento."""
    if not PREPROCESSOR_FILE.exists():
        warnings.warn(
            "Pré-processador não encontrado. Execute primeiro "
            "'python -m src.preparar_dados'.",
            stacklevel=2,
        )
        return None

    return joblib.load(PREPROCESSOR_FILE)


# ============================================================================
# 4. CARREGAMENTO DOS DADOS
# ============================================================================

def carregar_dados():
    """Carrega a base tratada e, quando disponível, a base integrada."""
    if not CLEAN_FILE.exists():
        raise FileNotFoundError(
            f"Base tratada não encontrada: {CLEAN_FILE}"
        )

    print("\nCarregando base tratada...")
    dados = pd.read_csv(CLEAN_FILE)

    dados_geograficos = None

    if INTEGRATED_FILE.exists():
        print("Carregando base integrada...")
        dados_geograficos = pd.read_csv(INTEGRATED_FILE)
    else:
        warnings.warn(
            "Base integrada não encontrada. "
            "O gráfico geográfico não será gerado.",
            stacklevel=2,
        )

    print(f"Registros carregados: {len(dados):,}")
    print(f"Variáveis disponíveis: {len(dados.columns)}")

    return dados, dados_geograficos


# ============================================================================
# 5. HISTOGRAMAS DAS VARIÁVEIS
# ============================================================================

def gerar_histogramas(dados):
    dados = preparar_dados(dados)

    figura, eixos = plt.subplots(
        5, 2,
        figsize=(16, 20),
    )

    for coluna, eixo in zip(FEATURES, eixos.flatten()):
        valores = dados[coluna].dropna()

        if valores.empty:
            eixo.set_visible(False)
            continue

        usar_kde = valores.std() > LIMIAR_VARIANCIA

        sns.histplot(
            x=valores,
            bins=30,
            kde=usar_kde,
            ax=eixo,
        )

        eixo.set(
            title=f"Distribuição de {coluna}",
            xlabel=coluna,
            ylabel="Quantidade de clientes",
        )

    figura.suptitle(
        "Distribuição das variáveis dos clientes",
        fontsize=16,
        fontweight="bold",
    )

    salvar_grafico(figura, "01_histogramas.png")


# ============================================================================
# 6. BOXPLOTS DAS VARIÁVEIS
# ============================================================================

def gerar_boxplots(dados):
    dados = preparar_dados(dados)

    figura, eixos = plt.subplots(
        5, 2,
        figsize=(16, 20),
    )

    for coluna, eixo in zip(FEATURES, eixos.flatten()):
        valores = dados[coluna].dropna()

        if valores.empty:
            eixo.set_visible(False)
            continue

        sns.boxplot(
            x=valores,
            ax=eixo,
        )

        eixo.set(
            title=f"Boxplot de {coluna}",
            xlabel=coluna,
        )

    figura.suptitle(
        "Identificação de valores extremos",
        fontsize=16,
        fontweight="bold",
    )

    salvar_grafico(figura, "02_boxplots.png")


# ============================================================================
# 7. GRÁFICOS DE DENSIDADE KDE
# ============================================================================

def gerar_densidades(dados):
    dados = preparar_dados(dados)

    figura, eixos = plt.subplots(
        5, 2,
        figsize=(16, 20),
    )

    for coluna, eixo in zip(FEATURES, eixos.flatten()):
        valores = dados[coluna].dropna()

        if (
            valores.empty
            or valores.nunique() < 2
            or valores.std() <= LIMIAR_VARIANCIA
        ):
            eixo.text(
                0.5, 0.5,
                "Sem variação suficiente para KDE",
                ha="center",
                va="center",
                transform=eixo.transAxes,
            )
            eixo.set_title(f"Densidade de {coluna}")
            continue

        sns.kdeplot(
            x=valores,
            fill=True,
            warn_singular=False,
            ax=eixo,
        )

        eixo.set(
            title=f"Densidade de {coluna}",
            xlabel=coluna,
            ylabel="Densidade",
        )

    figura.suptitle(
        "Distribuição de densidade das variáveis",
        fontsize=16,
        fontweight="bold",
    )

    salvar_grafico(figura, "03_densidades_kde.png")


# ============================================================================
# 8. MATRIZ DE CORRELAÇÃO
# ============================================================================

def gerar_matriz_correlacao(dados):
    dados = preparar_dados(dados)
    correlacao = dados.corr(method="pearson")

    figura, eixo = plt.subplots(figsize=(14, 11))

    sns.heatmap(
        correlacao,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        center=0,
        vmin=-1,
        vmax=1,
        ax=eixo,
    )

    eixo.set_title(
        "Matriz de correlação entre as variáveis",
        fontsize=15,
        fontweight="bold",
    )

    salvar_grafico(figura, "04_matriz_correlacao.png")


# ============================================================================
# 9. MATRIZ DE DISPERSÃO (PAIRPLOT)
# ============================================================================

def gerar_pairplot(dados):
    dados = preparar_dados(dados).dropna()

    if dados.empty:
        warnings.warn(
            "Não existem registros completos para gerar o Pairplot.",
            stacklevel=2,
        )
        return

    # Variáveis constantes não contribuem para a matriz de dispersão.
    colunas_variaveis = [
        coluna for coluna in FEATURES
        if dados[coluna].nunique() > 1
    ]

    if len(colunas_variaveis) < 2:
        warnings.warn(
            "Menos de duas variáveis apresentam variação. "
            "O Pairplot não será gerado.",
            stacklevel=2,
        )
        return

    dados = obter_amostra(dados[colunas_variaveis])

    grafico = sns.pairplot(
        dados,
        diag_kind="hist",
        corner=True,
        plot_kws={
            "alpha": 0.4,
            "s": 15,
        },
    )

    grafico.figure.suptitle(
        "Matriz de dispersão das variáveis",
        y=1.02,
        fontsize=15,
        fontweight="bold",
    )

    caminho = GRAFICOS_DIR / "05_pairplot.png"
    grafico.figure.savefig(
        caminho,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(grafico.figure)

    print("Gráfico gerado: 05_pairplot.png")


# ============================================================================
# 10. GRÁFICOS DE DISPERSÃO
# ============================================================================

def gerar_dispersoes(dados):
    dados = preparar_dados(dados)

    pares = [
        ("recencia", "frequencia"),
        ("frequencia", "valor_total"),
        ("ticket_medio", "frequencia"),
        ("valor_total", "frete_medio"),
        ("recencia", "valor_total"),
    ]

    figura, eixos = plt.subplots(
        3, 2,
        figsize=(16, 17),
    )

    for (eixo_x, eixo_y), eixo in zip(pares, eixos.flatten()):
        amostra = dados[[eixo_x, eixo_y]].dropna()

        if amostra.empty:
            eixo.set_visible(False)
            continue

        amostra = obter_amostra(amostra)

        sns.scatterplot(
            data=amostra,
            x=eixo_x,
            y=eixo_y,
            alpha=0.5,
            s=25,
            ax=eixo,
        )

        eixo.set_title(f"{eixo_x} × {eixo_y}")

    eixos.flatten()[-1].set_visible(False)

    figura.suptitle(
        "Relacionamento entre variáveis dos clientes",
        fontsize=16,
        fontweight="bold",
    )

    salvar_grafico(figura, "06_dispersoes.png")


# ============================================================================
# 11. GRÁFICO DE VALORES AUSENTES
# ============================================================================

def gerar_valores_ausentes(dados):
    dados = preparar_dados(dados)

    resultado = pd.DataFrame({
        "Quantidade": dados.isna().sum(),
        "Percentual": dados.isna().mean() * 100,
    }).sort_values(
        "Percentual",
        ascending=False,
    )

    figura, eixo = plt.subplots(figsize=(12, 7))

    sns.barplot(
        x=resultado.index,
        y=resultado["Percentual"],
        ax=eixo,
    )

    for indice, valor in enumerate(resultado["Percentual"]):
        eixo.text(
            indice,
            valor,
            f"{valor:.2f}%",
            ha="center",
            va="bottom",
        )

    eixo.set(
        title="Percentual de valores ausentes por variável",
        xlabel="Variáveis",
        ylabel="Valores ausentes (%)",
    )
    eixo.tick_params(axis="x", rotation=45)

    salvar_grafico(figura, "07_valores_ausentes.png")


# ============================================================================
# 12. MAPA DE AUSÊNCIA DE DADOS
# ============================================================================

def gerar_mapa_ausencias(dados):
    dados = obter_amostra(preparar_dados(dados))

    figura, eixo = plt.subplots(figsize=(14, 8))

    sns.heatmap(
        dados.isna(),
        cbar=False,
        yticklabels=False,
        cmap="viridis",
        ax=eixo,
    )

    eixo.set(
        title="Mapa de ausência de dados",
        xlabel="Variáveis",
        ylabel="Registros",
    )

    salvar_grafico(figura, "08_mapa_ausencias.png")


# ============================================================================
# 13. DISTRIBUIÇÃO DE AVALIAÇÕES
# ============================================================================

def gerar_distribuicao_avaliacoes(dados):
    dados = preparar_dados(dados)
    coluna = "possui_avaliacao"

    valores = dados[coluna].dropna()

    if valores.empty:
        warnings.warn(
            "Não existem dados para possui_avaliacao.",
            stacklevel=2,
        )
        return

    distribuicao = valores.value_counts().sort_index()

    rotulos = {
        0: "Sem avaliação",
        1: "Com avaliação",
    }
    nomes = [
        rotulos.get(valor, str(valor))
        for valor in distribuicao.index
    ]

    figura, eixo = plt.subplots(figsize=(9, 6))

    sns.barplot(
        x=nomes,
        y=distribuicao.values,
        ax=eixo,
    )

    for indice, valor in enumerate(distribuicao.values):
        eixo.text(
            indice,
            valor,
            f"{valor:,}".replace(",", "."),
            ha="center",
            va="bottom",
        )

    eixo.set(
        title="Distribuição de clientes com avaliação",
        xlabel="Situação da avaliação",
        ylabel="Quantidade de clientes",
    )

    salvar_grafico(figura, "09_distribuicao_avaliacoes.png")


# ============================================================================
# 14. DISTRIBUIÇÃO GEOGRÁFICA POR ESTADO
# ============================================================================

def gerar_distribuicao_geografica(dados_integrados):
    if dados_integrados is None:
        warnings.warn(
            "Base integrada indisponível. Gráfico geográfico ignorado.",
            stacklevel=2,
        )
        return

    coluna_estado = "estado"

    colunas_necessarias = [ID_COLUMN, coluna_estado]
    colunas_ausentes = [
        coluna for coluna in colunas_necessarias
        if coluna not in dados_integrados.columns
    ]

    if colunas_ausentes:
        warnings.warn(
            f"Colunas necessárias ao gráfico geográfico ausentes: "
            f"{colunas_ausentes}",
            stacklevel=2,
        )
        return

    dados_geo = (
        dados_integrados[[ID_COLUMN, coluna_estado]]
        .dropna(subset=[ID_COLUMN, coluna_estado])
        .drop_duplicates(subset=[ID_COLUMN])
    )

    clientes_estado = (
        dados_geo[coluna_estado]
        .astype(str)
        .str.strip()
        .value_counts()
        .sort_values(ascending=False)
    )

    if clientes_estado.empty:
        warnings.warn(
            "Não existem dados geográficos válidos.",
            stacklevel=2,
        )
        return

    figura, eixo = plt.subplots(figsize=(12, 6))

    sns.barplot(
        x=clientes_estado.index,
        y=clientes_estado.values,
        ax=eixo,
    )

    eixo.set(
        title="Distribuição de clientes por estado",
        xlabel="Estado",
        ylabel="Quantidade de clientes",
    )
    eixo.tick_params(axis="x", rotation=45)

    salvar_grafico(figura, "10_distribuicao_geografica.png")


# ============================================================================
# 15. COMPARAÇÃO ANTES E DEPOIS DA PADRONIZAÇÃO
# ============================================================================

def gerar_comparacao_padronizacao(dados_originais):
    """
    Compara os valores originais e transformados do conjunto de treinamento.

    A divisão é reproduzida com os mesmos parâmetros do projeto. O
    pré-processador salvo é apenas aplicado com transform(), sem novo ajuste.
    """
    print("\nCarregando pré-processador do projeto...")

    preprocessor = carregar_preprocessador()
    if preprocessor is None:
        return

    X = preparar_dados(dados_originais)

    X_treino, _ = train_test_split(
        X,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    valores_transformados = preprocessor.transform(X_treino)

    if valores_transformados.shape[1] != len(FEATURES):
        raise ValueError(
            "A quantidade de colunas produzidas pelo pré-processador "
            "não corresponde à quantidade de FEATURES."
        )

    dados_padronizados = pd.DataFrame(
        valores_transformados,
        columns=FEATURES,
        index=X_treino.index,
    )

    # Para comparar as distribuições sem misturar valores imputados,
    # a visualização utiliza apenas registros completos no conjunto original.
    indices_validos = X_treino.dropna().index

    dados_originais_plot = X_treino.loc[indices_validos]
    dados_padronizados_plot = dados_padronizados.loc[indices_validos]

    if dados_originais_plot.empty:
        warnings.warn(
            "Não existem registros completos para comparar a padronização.",
            stacklevel=2,
        )
        return

    figura, eixos = plt.subplots(
        5, 2,
        figsize=(16, 20),
    )

    for coluna, eixo in zip(FEATURES, eixos.flatten()):
        serie_original = dados_originais_plot[coluna].astype(float)
        serie_padronizada = dados_padronizados_plot[coluna].astype(float)

        variacao_original = (
            len(serie_original) > 1
            and serie_original.std() > LIMIAR_VARIANCIA
        )
        variacao_padronizada = (
            len(serie_padronizada) > 1
            and serie_padronizada.std() > LIMIAR_VARIANCIA
        )

        if variacao_original:
            sns.kdeplot(
                x=serie_original,
                label="Original",
                warn_singular=False,
                ax=eixo,
            )
        elif not serie_original.empty:
            eixo.axvline(
                serie_original.median(),
                label="Original (sem variação)",
                linestyle="--",
            )

        if variacao_padronizada:
            sns.kdeplot(
                x=serie_padronizada,
                label="Padronizado",
                warn_singular=False,
                ax=eixo,
            )
        elif not serie_padronizada.empty:
            eixo.axvline(
                serie_padronizada.median(),
                label="Padronizado (sem variação)",
                linestyle=":",
            )

        eixo.set(
            title=f"Comparação: {coluna}",
            xlabel=coluna,
            ylabel="Densidade",
        )

        handles, labels = eixo.get_legend_handles_labels()
        if handles:
            eixo.legend(handles, labels)

    figura.suptitle(
        "Distribuição antes e depois da padronização",
        fontsize=16,
        fontweight="bold",
    )

    salvar_grafico(
        figura,
        "11_comparacao_padronizacao.png",
    )


# ============================================================================
# 16. PROJEÇÃO BIDIMENSIONAL COM PCA
# ============================================================================

def gerar_pca(dados):
    """
    Gera uma projeção exploratória em duas dimensões.

    Usa o pré-processador do projeto para manter a mesma transformação das
    variáveis. A PCA desta função é apenas exploratória e não treina K-Means
    ou DBSCAN.
    """
    X = preparar_dados(dados)

    preprocessor = carregar_preprocessador()
    if preprocessor is None:
        return

    dados_transformados = preprocessor.transform(X)

    if len(X) < 2 or dados_transformados.shape[1] < 2:
        warnings.warn(
            "Quantidade insuficiente de registros ou variáveis para PCA.",
            stacklevel=2,
        )
        return

    pca = PCA(
        n_components=2,
        random_state=RANDOM_STATE,
    )

    componentes = pca.fit_transform(dados_transformados)

    resultado = pd.DataFrame(
        componentes,
        columns=[
            "Componente Principal 1",
            "Componente Principal 2",
        ],
    )

    variancia = pca.explained_variance_ratio_

    figura, eixo = plt.subplots(figsize=(11, 8))

    sns.scatterplot(
        data=resultado,
        x="Componente Principal 1",
        y="Componente Principal 2",
        alpha=0.5,
        s=25,
        ax=eixo,
    )

    eixo.set(
        title="Projeção bidimensional dos clientes - PCA",
        xlabel=f"PC1 ({variancia[0] * 100:.2f}% da variância)",
        ylabel=f"PC2 ({variancia[1] * 100:.2f}% da variância)",
    )

    salvar_grafico(figura, "12_pca_bidimensional.png")


# ============================================================================
# 17. FUNÇÃO PRINCIPAL DA ANÁLISE EXPLORATÓRIA
# ============================================================================

def analisar_dados():
    print("\n" + "=" * 70)
    print("PROJETO OLIST - ANÁLISE EXPLORATÓRIA DE DADOS")
    print("=" * 70)

    preparar_diretorios()

    dados, dados_geograficos = carregar_dados()

    print("\nPreparando variáveis...")
    dados_numericos = preparar_dados(dados)

    print("\nResumo da base:")
    print(f"Quantidade de registros: {len(dados_numericos):,}")
    print(f"Quantidade de variáveis: {len(FEATURES)}")

    print("\nGerando gráficos...")

    gerar_histogramas(dados_numericos)
    gerar_boxplots(dados_numericos)
    gerar_densidades(dados_numericos)
    gerar_matriz_correlacao(dados_numericos)
    gerar_pairplot(dados_numericos)
    gerar_dispersoes(dados_numericos)
    gerar_valores_ausentes(dados_numericos)
    gerar_mapa_ausencias(dados_numericos)
    gerar_distribuicao_avaliacoes(dados_numericos)
    gerar_distribuicao_geografica(dados_geograficos)

    print("\nComparando dados antes e depois da padronização...")
    gerar_comparacao_padronizacao(dados)

    print("\nGerando projeção PCA...")
    gerar_pca(dados_numericos)

    print("\n" + "=" * 70)
    print("ANÁLISE EXPLORATÓRIA FINALIZADA")
    print("=" * 70)
    print(f"\nGráficos armazenados em: {GRAFICOS_DIR}")
    print("Total de arquivos previstos: 12")


# ============================================================================
# 18. EXECUÇÃO DO MÓDULO
# ============================================================================

if __name__ == "__main__":
    analisar_dados()
