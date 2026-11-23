
import joblib
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors

from src.config import (
    TEST_FILE,
    TEST_IDS_FILE,
    KMEANS_FILE,
    DBSCAN_CORE_FILE,
    PREDICTIONS_FILE,
    REPORTS_DIR,
    create_directories
)


def predizer_dbscan(X, core):

    pontos_centrais = core["core_samples"]
    rotulos_centrais = core["core_labels"]
    eps = core["eps"]

    if len(pontos_centrais) == 0:
        return np.full(len(X), -1, dtype=int)

    vizinhos = NearestNeighbors(
        n_neighbors=1,
        n_jobs=1
    )

    vizinhos.fit(pontos_centrais)

    distancias, indices = vizinhos.kneighbors(X)

    rotulos = np.where(
        distancias[:, 0] <= eps,
        rotulos_centrais[indices[:, 0]],
        -1
    )

    return rotulos


def predizer_clientes():

    create_directories()

    arquivos = [
        TEST_FILE,
        TEST_IDS_FILE,
        KMEANS_FILE,
        DBSCAN_CORE_FILE
    ]

    for arquivo in arquivos:
        if not arquivo.exists():
            raise FileNotFoundError(
                f"Arquivo necessário não encontrado: {arquivo}"
            )

    # Carrega os dados

    X = pd.read_csv(
        TEST_FILE
    ).to_numpy(
        dtype=np.float32
    )

    ids = pd.read_csv(
        TEST_IDS_FILE,
        dtype={"customer_unique_id": "string"}
    )

    if len(X) != len(ids):
        raise ValueError(
            "Quantidade de registros e identificadores incompatível."
        )

    # Carrega os modelos

    kmeans = joblib.load(KMEANS_FILE)

    core = joblib.load(DBSCAN_CORE_FILE)

    # Predições

    labels_kmeans = kmeans.predict(X)

    labels_dbscan = predizer_dbscan(
        X,
        core
    )

    # Resultado final

    resultado = pd.DataFrame({
        "customer_unique_id": ids["customer_unique_id"],
        "cluster_kmeans": labels_kmeans,
        "cluster_dbscan": labels_dbscan,
        "dbscan_ruido": (
            labels_dbscan == -1
        ).astype(int)
    })

    # Salva resultados

    resultado.to_csv(
        PREDICTIONS_FILE,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n"
    )

    # Resumo dos agrupamentos

    resumo_kmeans = (
        resultado["cluster_kmeans"]
        .value_counts()
        .sort_index()
    )

    resumo_dbscan = (
        resultado["cluster_dbscan"]
        .value_counts()
        .sort_index()
    )

    resumo = pd.DataFrame({
        "KMeans": resumo_kmeans,
        "DBSCAN": resumo_dbscan
    })

    resumo.to_csv(
        REPORTS_DIR / "resumo_predicoes.csv",
        encoding="utf-8-sig"
    )

    print("\nPREDIÇÃO CONCLUÍDA")

    print("\nPrimeiros resultados:")
    print(resultado.head(10))

    print("\nDistribuição K-Means:")
    print(resumo_kmeans)

    print("\nDistribuição DBSCAN:")
    print(resumo_dbscan)

    print(
        "\nPercentual de ruído DBSCAN:",
        f"{(labels_dbscan == -1).mean():.2%}"
    )

    print("\nArquivo gerado:")
    print(PREDICTIONS_FILE)


if __name__ == "__main__":
    predizer_clientes()
