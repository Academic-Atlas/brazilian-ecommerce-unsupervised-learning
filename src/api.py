
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import (
    PREPROCESSOR_FILE,
    KMEANS_FILE,
    FEATURES
)


app = FastAPI(
    title="API de Segmentação de Clientes Olist",
    description="Classifica clientes utilizando o modelo K-Means.",
    version="1.0.0"
)


class Cliente(BaseModel):
    recencia: float
    frequencia: float
    valor_total: float
    ticket_medio: float
    total_itens: float
    valor_produtos_total: float
    frete_medio: float
    avaliacao_media: float | None = None
    possui_avaliacao: float = 1
    parcelas_media: float | None = None


@app.get("/")
def inicio():
    return {
        "mensagem": "API de segmentação Olist. Consulte /docs."
    }


@app.post("/segmentar")
def segmentar(cliente: Cliente):

    # Verifica os arquivos necessários
    if not PREPROCESSOR_FILE.exists():
        raise HTTPException(
            status_code=503,
            detail="Pré-processador não encontrado."
        )

    if not KMEANS_FILE.exists():
        raise HTTPException(
            status_code=503,
            detail="Modelo K-Means não encontrado."
        )

    try:
        # Carrega os artefatos
        preprocessador = joblib.load(PREPROCESSOR_FILE)
        modelo = joblib.load(KMEANS_FILE)

        # Converte os dados recebidos em DataFrame
        dados = pd.DataFrame(
            [cliente.model_dump()]
        )[FEATURES]

        dados = dados.apply(
            pd.to_numeric,
            errors="coerce"
        )

        dados = dados.replace(
            [np.inf, -np.inf],
            np.nan
        )

        # Aplica o mesmo pré-processamento do treinamento
        X = preprocessador.transform(dados)

        # Ajusta o tipo numérico ao utilizado pelo modelo K-Means
        X = np.asarray(
            X,
        dtype=modelo.cluster_centers_.dtype,
        order="C"
)

        # Realiza a predição
        grupo = int(modelo.predict(X)[0])

        return {
            "algoritmo": "K-Means",
            "grupo_cliente": grupo,
            "observacao": (
                "O número do grupo é um identificador técnico. "
                "Consulte o perfil dos clusters para sua "
                "interpretação comercial."
            )
        }

    except Exception as erro:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao realizar a segmentação: {str(erro)}"
        )

