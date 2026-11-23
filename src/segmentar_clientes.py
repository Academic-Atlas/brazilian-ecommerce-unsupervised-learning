
import gc
import joblib
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors

from src.config import (
    CLEAN_FILE,
    PREPROCESSOR_FILE,
    KMEANS_FILE,
    DBSCAN_CORE_FILE,
    FEATURES,
    DATA_PROCESSED,
    create_directories
)


def carregar_dados_e_modelos():

    create_directories()

    arquivos = [
        CLEAN_FILE,
        PREPROCESSOR_FILE,
        KMEANS_FILE,
        DBSCAN_CORE_FILE
    ]

    for arquivo in arquivos:
        if not arquivo.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {arquivo}"
            )

    print("Carregando base tratada...")

    dados = pd.read_csv(CLEAN_FILE)

    print("Carregando pré-processador...")
    preprocessor = joblib.load(PREPROCESSOR_FILE)

    print("Carregando K-Means...")
    kmeans = joblib.load(KMEANS_FILE)

    print("Carregando pontos centrais do DBSCAN...")
    dbscan_core = joblib.load(DBSCAN_CORE_FILE)

    return dados, preprocessor, kmeans, dbscan_core


def preparar_matriz(dados, preprocessor):

    X = dados[FEATURES].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_scaled = preprocessor.transform(X)

    return np.asarray(
        X_scaled,
        dtype=np.float32
    )


def segmentar_kmeans(dados, X_scaled, modelo):

    print("\nSEGMENTANDO COM K-MEANS...")

    resultado = dados.copy()

    resultado["cluster"] = modelo.predict(X_scaled)

    saida = DATA_PROCESSED / "clientes_segmentados_kmeans.csv"

    resultado.to_csv(
        saida,
        index=False,
        encoding="utf-8-sig"
    )

    perfil = resultado.groupby("cluster")[FEATURES].agg(
        ["count", "mean", "median"]
    )

    perfil_path = DATA_PROCESSED / "perfil_clusters_kmeans.csv"

    perfil.to_csv(
        perfil_path,
        encoding="utf-8-sig"
    )

    print("Clientes segmentados:", len(resultado))
    print("Quantidade de clusters:", resultado["cluster"].nunique())
    print("Arquivo:", saida)
    print("\nPERFIL DOS CLUSTERS K-MEANS")
    print(perfil)

    return resultado


def segmentar_dbscan(dados, X_scaled, dbscan_core):

    print("\nSEGMENTANDO COM DBSCAN...")

    core_samples = dbscan_core["core_samples"]
    core_labels = dbscan_core["core_labels"]
    eps = dbscan_core["eps"]

    if len(core_samples) == 0:
        raise ValueError(
            "O DBSCAN não possui pontos centrais disponíveis."
        )

    resultado = dados.copy()

    # Inicialmente todos são considerados ruído

    labels = np.full(
        len(X_scaled),
        -1,
        dtype=int
    )

    # Busca o ponto central mais próximo
    # Processamento em lotes para economizar memória

    vizinhos = NearestNeighbors(
        n_neighbors=1,
        algorithm="brute",
        n_jobs=1
    )

    vizinhos.fit(core_samples)

    tamanho_lote = 1000

    for inicio in range(0, len(X_scaled), tamanho_lote):

        fim = min(
            inicio + tamanho_lote,
            len(X_scaled)
        )

        lote = X_scaled[inicio:fim]

        distancias, indices = vizinhos.kneighbors(
            lote,
            return_distance=True
        )

        distancias = distancias.ravel()
        indices = indices.ravel()

        dentro_raio = distancias <= eps

        labels_lote = np.full(
            len(lote),
            -1,
            dtype=int
        )

        labels_lote[dentro_raio] = core_labels[
            indices[dentro_raio]
        ]

        labels[inicio:fim] = labels_lote

        print(
            f"Processados: {fim}/{len(X_scaled)}",
            end="\r"
        )

    resultado["cluster"] = labels

    saida = DATA_PROCESSED / "clientes_segmentados_dbscan.csv"

    resultado.to_csv(
        saida,
        index=False,
        encoding="utf-8-sig"
    )

    perfil = resultado.groupby("cluster")[FEATURES].agg(
        ["count", "mean", "median"]
    )

    perfil_path = DATA_PROCESSED / "perfil_clusters_dbscan.csv"

    perfil.to_csv(
        perfil_path,
        encoding="utf-8-sig"
    )

    taxa_ruido = np.mean(labels == -1)

    print("\n\nClientes segmentados:", len(resultado))
    print("Clusters encontrados:", len(set(labels) - {-1}))
    print(f"Taxa de ruído: {taxa_ruido:.2%}")
    print("Arquivo:", saida)
    print("\nPERFIL DOS CLUSTERS DBSCAN")
    print(perfil)

    return resultado


def executar_segmentacao():

    dados, preprocessor, kmeans, dbscan_core = (
        carregar_dados_e_modelos()
    )

    print("\nPreparando variáveis...")

    X_scaled = preparar_matriz(
        dados,
        preprocessor
    )

    print("Dimensões:", X_scaled.shape)

    resultado_kmeans = segmentar_kmeans(
        dados,
        X_scaled,
        kmeans
    )

    gc.collect()

    resultado_dbscan = segmentar_dbscan(
        dados,
        X_scaled,
        dbscan_core
    )

    print("\nSEGMENTAÇÃO CONCLUÍDA")

    print(
        "Consumidores analisados:",
        len(dados)
    )

    print(
        "K-Means:",
        len(resultado_kmeans)
    )

    print(
        "DBSCAN:",
        len(resultado_dbscan)
    )


if __name__ == "__main__":
    executar_segmentacao()

