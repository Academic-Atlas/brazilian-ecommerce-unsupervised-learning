
import gc
import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

from src.config import (
    TRAIN_FILE,
    KMEANS_FILE,
    DBSCAN_FILE,
    DBSCAN_CORE_FILE,
    K_MIN,
    K_MAX,
    RANDOM_STATE,
    SILHOUETTE_SAMPLE_SIZE,
    DBSCAN_SAMPLE_SIZE,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    REPORTS_DIR,
    create_directories
)


def calcular_silhouette(X, labels):

    labels = np.asarray(labels)

    # Exclui ruídos do DBSCAN

    mascara = labels != -1

    X_valido = X[mascara]
    labels_validos = labels[mascara]

    grupos = np.unique(labels_validos)

    if len(grupos) < 2:
        return np.nan

    if len(X_valido) <= len(grupos):
        return np.nan

    try:

        if len(X_valido) > SILHOUETTE_SAMPLE_SIZE:

            return silhouette_score(
                X_valido,
                labels_validos,
                sample_size=SILHOUETTE_SAMPLE_SIZE,
                random_state=RANDOM_STATE
            )

        return silhouette_score(
            X_valido,
            labels_validos
        )

    except Exception:
        return np.nan


def treinar_kmeans(X):

    print("\nINICIANDO K-MEANS")

    resultados = []

    for k in range(K_MIN, K_MAX + 1):

        modelo = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10
        )

        labels = modelo.fit_predict(X)

        silhouette = calcular_silhouette(
            X,
            labels
        )

        resultados.append({
            "k": k,
            "inercia": modelo.inertia_,
            "silhouette": silhouette
        })

        print(
            f"K={k} | "
            f"Inércia={modelo.inertia_:.2f} | "
            f"Silhouette={silhouette}"
        )

        del modelo, labels
        gc.collect()

    resultados_df = pd.DataFrame(resultados)

    resultados_df.to_csv(
        REPORTS_DIR / "experimentos_kmeans.csv",
        index=False,
        encoding="utf-8-sig"
    )

    validos = resultados_df.dropna(
        subset=["silhouette"]
    )

    if validos.empty:
        raise ValueError(
            "Não foi possível selecionar K por Silhouette."
        )

    melhor_k = int(
        validos.sort_values(
            "silhouette",
            ascending=False
        ).iloc[0]["k"]
    )

    print("\nMelhor K encontrado:", melhor_k)

    modelo_final = KMeans(
        n_clusters=melhor_k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    labels_final = modelo_final.fit_predict(X)

    joblib.dump(
        modelo_final,
        KMEANS_FILE
    )

    pd.DataFrame({
        "cluster": labels_final
    }).to_csv(
        REPORTS_DIR / "clusters_kmeans_treino.csv",
        index=False
    )

    print("K-Means salvo:", KMEANS_FILE)

    return modelo_final


def treinar_dbscan(X):

    print("\nINICIANDO DBSCAN")

    # Amostragem reproduzível

    if len(X) > DBSCAN_SAMPLE_SIZE:

        rng = np.random.default_rng(RANDOM_STATE)

        indices = rng.choice(
            len(X),
            size=DBSCAN_SAMPLE_SIZE,
            replace=False
        )

        X_amostra = X[indices]

    else:
        X_amostra = X

    # Parâmetros exploratórios

    eps_candidatos = [
        0.8, 1.0, 1.2, 1.5, 1.8, 2.0
    ]

    resultados = []
    melhor_eps = None
    melhor_silhouette = -np.inf

    for eps in eps_candidatos:

        modelo = DBSCAN(
            eps=eps,
            min_samples=DBSCAN_MIN_SAMPLES,
            n_jobs=1
        )

        labels = modelo.fit_predict(X_amostra)

        clusters = len(
            set(labels) - {-1}
        )

        taxa_ruido = np.mean(
            labels == -1
        )

        silhouette = calcular_silhouette(
            X_amostra,
            labels
        )

        resultados.append({
            "eps": eps,
            "min_samples": DBSCAN_MIN_SAMPLES,
            "clusters": clusters,
            "taxa_ruido": taxa_ruido,
            "silhouette": silhouette
        })

        print(
            f"eps={eps} | "
            f"Clusters={clusters} | "
            f"Ruído={taxa_ruido:.2%} | "
            f"Silhouette={silhouette}"
        )

        if (
            clusters >= 2
            and np.isfinite(silhouette)
            and silhouette > melhor_silhouette
        ):

            melhor_silhouette = silhouette
            melhor_eps = eps

        del modelo, labels
        gc.collect()

    resultados_df = pd.DataFrame(resultados)

    resultados_df.to_csv(
        REPORTS_DIR / "experimentos_dbscan.csv",
        index=False,
        encoding="utf-8-sig"
    )

    if melhor_eps is None:

        print(
            "Nenhum eps produziu pelo menos dois clusters "
            "com Silhouette válida. Será utilizado o eps "
            "configurado como alternativa exploratória."
        )

        melhor_eps = DBSCAN_EPS

    print("\nMelhor eps encontrado:", melhor_eps)

    # Modelo final na amostra

    modelo_final = DBSCAN(
        eps=melhor_eps,
        min_samples=DBSCAN_MIN_SAMPLES,
        n_jobs=1
    )

    labels_final = modelo_final.fit_predict(
        X_amostra
    )

    joblib.dump(
        modelo_final,
        DBSCAN_FILE
    )

    # Salva os pontos centrais e seus rótulos.
    # DBSCAN não possui método predict nativo.

    core = {
        "core_samples": modelo_final.components_.astype(
            np.float32
        ),
        "core_labels": labels_final[
            modelo_final.core_sample_indices_
        ],
        "eps": melhor_eps,
        "min_samples": DBSCAN_MIN_SAMPLES,
        "quantidade_treinamento": len(X_amostra)
    }

    joblib.dump(
        core,
        DBSCAN_CORE_FILE
    )

    pd.DataFrame({
        "cluster": labels_final
    }).to_csv(
        REPORTS_DIR / "clusters_dbscan_treino.csv",
        index=False
    )

    print("DBSCAN salvo:", DBSCAN_FILE)
    print("Pontos centrais salvos:", DBSCAN_CORE_FILE)
    print("Taxa de ruído:", np.mean(labels_final == -1))

    return modelo_final


def treinar_modelos():

    create_directories()

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            "Execute primeiro: python -m src.preparar_dados"
        )

    X = pd.read_csv(
        TRAIN_FILE
    ).to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(X).all():
        raise ValueError(
            "Existem valores infinitos ou ausentes no treino."
        )

    print("Dimensões do treinamento:", X.shape)

    treinar_kmeans(X)

    gc.collect()

    treinar_dbscan(X)

    print("\nTREINAMENTO CONCLUÍDO")


if __name__ == "__main__":
    treinar_modelos()
