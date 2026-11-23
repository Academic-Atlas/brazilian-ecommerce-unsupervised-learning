
import numpy as np
import pandas as pd

from src.config import (
    INTEGRATED_FILE,
    CLEAN_FILE,
    FEATURES,
    ID_COLUMN,
    REPORTS_DIR,
    create_directories
)


def tratar_dados():

    create_directories()

    if not INTEGRATED_FILE.exists():
        raise FileNotFoundError(
            "Execute primeiro: python -m src.integrar_dados"
        )

    print("Lendo base integrada...")

    dados = pd.read_csv(
        INTEGRATED_FILE,
        parse_dates=["ultima_compra"],
        dtype={ID_COLUMN: "string"}
    )

    if dados.shape[1] != 14:
        raise ValueError(
            f"Esperadas 14 colunas; encontradas {dados.shape[1]}."
        )

    if dados[ID_COLUMN].isna().any():
        raise ValueError("Existem identificadores ausentes.")

    if dados[ID_COLUMN].duplicated().any():
        raise ValueError("Existem consumidores duplicados.")

    ids_validos = dados[ID_COLUMN].str.fullmatch(
        r"[0-9a-fA-F]{32}"
    )

    if not ids_validos.all():
        raise ValueError(
            "Existem identificadores fora do padrão esperado."
        )

    linhas_iniciais = len(dados)

    # Conversão das variáveis numéricas

    for coluna in FEATURES:

        if coluna not in dados.columns:
            raise ValueError(
                f"Coluna obrigatória ausente: {coluna}"
            )

        dados[coluna] = pd.to_numeric(
            dados[coluna],
            errors="coerce"
        )

    # Tratamento de infinitos

    dados[FEATURES] = dados[FEATURES].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Variáveis que não podem assumir valores negativos

    nao_negativas = [
        "recencia",
        "frequencia",
        "valor_total",
        "ticket_medio",
        "total_itens",
        "valor_produtos_total",
        "frete_medio",
        "parcelas_media"
    ]

    negativos = {}

    for coluna in nao_negativas:

        negativos[coluna] = int(
            (dados[coluna] < 0).sum()
        )

        dados.loc[
            dados[coluna] < 0,
            coluna
        ] = np.nan

    # Frequência deve ser positiva e recência não negativa

    dados = dados.loc[
        dados["frequencia"].gt(0)
        & dados["recencia"].ge(0)
    ].copy()

    # Indicador de avaliação

    dados["possui_avaliacao"] = (
        dados["possui_avaliacao"]
        .fillna(0)
        .clip(0, 1)
    )

    dados = dados.sort_values(
        ID_COLUMN
    ).reset_index(drop=True)

    # Gravação e validação

    arquivo_temp = CLEAN_FILE.with_name(
        CLEAN_FILE.stem + "_temporario.csv"
    )

    dados.to_csv(
        arquivo_temp,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n"
    )

    verificacao = pd.read_csv(
        arquivo_temp,
        dtype={ID_COLUMN: "string"}
    )

    if verificacao.shape != dados.shape:
        raise ValueError(
            "A validação do arquivo tratado falhou."
        )

    arquivo_temp.replace(CLEAN_FILE)

    # Relatório

    resumo = [
        "RESUMO DO TRATAMENTO",
        f"Registros iniciais: {linhas_iniciais}",
        f"Registros finais: {len(dados)}",
        f"Colunas: {len(dados.columns)}",
        f"Duplicidades: {dados[ID_COLUMN].duplicated().sum()}",
        "",
        "Valores negativos convertidos em ausentes:"
    ]

    resumo += [
        f"{coluna}: {quantidade}"
        for coluna, quantidade in negativos.items()
    ]

    resumo += [
        "",
        "Valores ausentes:",
        dados[FEATURES].isna().sum().to_string()
    ]

    (REPORTS_DIR / "resumo_tratamento.txt").write_text(
        "\n".join(resumo),
        encoding="utf-8"
    )

    print("\n".join(resumo))
    print("\nBase tratada:", CLEAN_FILE)


if __name__ == "__main__":
    tratar_dados()
