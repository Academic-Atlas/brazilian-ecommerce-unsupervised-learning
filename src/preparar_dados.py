
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.config import (
    CLEAN_FILE,
    TRAIN_FILE,
    TEST_FILE,
    TEST_IDS_FILE,
    PREPROCESSOR_FILE,
    FEATURES,
    ID_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    create_directories
)


def preparar_dados():

    create_directories()

    if not CLEAN_FILE.exists():
        raise FileNotFoundError(
            "Execute primeiro: python -m src.tratar_dados"
        )

    dados = pd.read_csv(
        CLEAN_FILE,
        dtype={ID_COLUMN: "string"}
    )

    X = dados[FEATURES].copy()
    ids = dados[ID_COLUMN].copy()

    X_train, X_test, ids_train, ids_test = train_test_split(
        X,
        ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # Pipeline de pré-processamento

    preprocessor = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    # Ajuste somente no treinamento

    X_train_scaled = preprocessor.fit_transform(X_train)

    # Teste recebe apenas a transformação

    X_test_scaled = preprocessor.transform(X_test)

    # Salva treinamento

    pd.DataFrame(
        X_train_scaled,
        columns=FEATURES
    ).to_csv(
        TRAIN_FILE,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n"
    )

    # Salva teste

    pd.DataFrame(
        X_test_scaled,
        columns=FEATURES
    ).to_csv(
        TEST_FILE,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n"
    )

    # Salva identificadores do teste

    pd.DataFrame({
        ID_COLUMN: ids_test.reset_index(drop=True)
    }).to_csv(
        TEST_IDS_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # Salva pré-processador

    joblib.dump(
        preprocessor,
        PREPROCESSOR_FILE
    )

    # Validação das gravações

    treino_check = pd.read_csv(TRAIN_FILE)
    teste_check = pd.read_csv(TEST_FILE)

    if treino_check.shape != X_train_scaled.shape:
        raise ValueError(
            "Arquivo de treinamento inconsistente."
        )

    if teste_check.shape != X_test_scaled.shape:
        raise ValueError(
            "Arquivo de teste inconsistente."
        )

    print("\nPREPARAÇÃO CONCLUÍDA")

    print("Total de clientes:", len(dados))
    print("Treinamento:", len(X_train_scaled))
    print("Teste:", len(X_test_scaled))

    print("\nArquivos preparados:")
    print(TRAIN_FILE)
    print(TEST_FILE)
    print(PREPROCESSOR_FILE)


if __name__ == "__main__":
    preparar_dados()
