
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

from sklearn.neighbors import NearestNeighbors

from src.config import (
    TEST_FILE,
    KMEANS_FILE,
    DBSCAN_CORE_FILE,
    METRICS_FILE,
    SILHOUETTE_SAMPLE_SIZE,
    RANDOM_STATE,
    create_directories
)


def avaliar_metricas(X, labels, nome):

    labels = np.asarray(labels)

    # Remove ruídos para calcular métricas de clusters

    mascara = labels != -1

    X_valido = X[mascara]
    labels_validos = labels[mascara]

    quantidade_clusters = len(
        np.unique(labels_validos)
    )

    resultado = {
        "algoritmo": nome,
        "registros": len(labels),
        "clusters": quantidade_clusters,
        "taxa_ruido": np.mean(labels == -1),
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan
    }

    if (
        quantidade_clusters < 2
        or len(X_valido) <= quantidade_clusters
    ):
        return resultado

    try:

        if len(X_valido) > SILHOUETTE_SAMPLE_SIZE:

            resultado["silhouette"] = silhouette_score(
                X_valido,
                labels_validos,
                sample_size=SILHOUETTE_SAMPLE_SIZE,
                random_state=RANDOM_STATE
            )

        else:

            resultado["silhouette"] = silhouette_score(
                X_valido,
                labels_validos
            )

        resultado["davies_bouldin"] = (
            davies_bouldin_score(
                X_valido,
                labels_validos
            )
        )

        resultado["calinski_harabasz"] = (
            calinski_harabasz_score(
                X_valido,
                labels_validos
            )
        )

    except Exception as erro:

        print(
            f"Erro ao calcular métricas de {nome}: {erro}"
        )

    return resultado


def predizer_dbscan(X, core):

    pontos_centrais = core["core_samples"]
    rotulos_centrais = core["core_labels"]
    eps = core["eps"]

    if len(pontos_centrais) == 0:

        return np.full(
            len(X),
            -1,
            dtype=int
        )

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


def avaliar_modelos():

    create_directories()

    X = pd.read_csv(
        TEST_FILE
    ).to_numpy(
        dtype=np.float32
    )

    kmeans = joblib.load(KMEANS_FILE)

    core = joblib.load(DBSCAN_CORE_FILE)

    # Predição K-Means

    labels_kmeans = kmeans.predict(X)

    # Atribuição aproximada DBSCAN

    labels_dbscan = predizer_dbscan(
        X,
        core
    )

    # Métricas

    resultados = []

    resultados.append(
        avaliar_metricas(
            X,
            labels_kmeans,
            "K-Means"
        )
    )

    resultados.append(
        avaliar_metricas(
            X,
            labels_dbscan,
            "DBSCAN_aproximado"
        )
    )

    metricas = pd.DataFrame(resultados)

    metricas.to_csv(
        METRICS_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nRESULTADOS DA AVALIAÇÃO")

    print(
        metricas.to_string(index=False)
    )

    print("\nMétricas salvas em:")
    print(METRICS_FILE)

    print(
        "\nAtenção: DBSCAN utiliza atribuição aproximada "
        "no conjunto de teste, não uma função predict nativa."
    )


if __name__ == "__main__":
    avaliar_modelos()
