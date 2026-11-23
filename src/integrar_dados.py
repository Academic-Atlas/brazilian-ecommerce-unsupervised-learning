
import pandas as pd

from src.config import (
    CUSTOMERS_FILE,
    ORDERS_FILE,
    ITEMS_FILE,
    PAYMENTS_FILE,
    REVIEWS_FILE,
    INTEGRATED_FILE,
    START_DATE,
    END_DATE,
    REFERENCE_DATE,
    create_directories
)


def integrar_dados():

    create_directories()

    arquivos = [
        CUSTOMERS_FILE,
        ORDERS_FILE,
        ITEMS_FILE,
        PAYMENTS_FILE,
        REVIEWS_FILE
    ]

    for arquivo in arquivos:
        if not arquivo.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {arquivo}"
            )

    print("Carregando arquivos originais...")

    clientes = pd.read_csv(
        CUSTOMERS_FILE,
        dtype={
            "customer_id": "string",
            "customer_unique_id": "string"
        }
    )

    pedidos = pd.read_csv(
        ORDERS_FILE,
        parse_dates=["order_purchase_timestamp"],
        dtype={
            "order_id": "string",
            "customer_id": "string"
        }
    )

    itens = pd.read_csv(
        ITEMS_FILE,
        dtype={"order_id": "string"}
    )

    pagamentos = pd.read_csv(
        PAYMENTS_FILE,
        dtype={"order_id": "string"}
    )

    avaliacoes = pd.read_csv(
        REVIEWS_FILE,
        dtype={"order_id": "string"}
    )

    print("Filtrando pedidos de 2017...")

    pedidos = pedidos.loc[
        pedidos["order_purchase_timestamp"].ge(
            pd.Timestamp(START_DATE)
        )
        & pedidos["order_purchase_timestamp"].lt(
            pd.Timestamp(END_DATE)
        )
        & pedidos["order_status"].eq("delivered")
    ].copy()

    if pedidos.empty:
        raise ValueError(
            "Nenhum pedido entregue encontrado em 2017."
        )

    # Agregação dos itens por pedido

    itens["price"] = pd.to_numeric(
        itens["price"], errors="coerce"
    )

    itens["freight_value"] = pd.to_numeric(
        itens["freight_value"], errors="coerce"
    )

    itens_agg = itens.groupby(
        "order_id", as_index=False
    ).agg(
        total_itens=("order_item_id", "count"),
        valor_produtos=("price", "sum"),
        valor_frete=("freight_value", "sum")
    )

    itens_agg["valor_pedido"] = (
        itens_agg["valor_produtos"]
        + itens_agg["valor_frete"]
    )

    # Agregação dos pagamentos

    pagamentos["payment_installments"] = pd.to_numeric(
        pagamentos["payment_installments"],
        errors="coerce"
    )

    pagamentos_agg = pagamentos.groupby(
        "order_id", as_index=False
    ).agg(
        parcelas_media=("payment_installments", "mean")
    )

    # Agregação das avaliações

    avaliacoes["review_score"] = pd.to_numeric(
        avaliacoes["review_score"],
        errors="coerce"
    )

    avaliacoes_agg = avaliacoes.groupby(
        "order_id", as_index=False
    ).agg(
        avaliacao_media=("review_score", "mean")
    )

    # Integração dos dados

    base = pedidos.merge(
        clientes[
            [
                "customer_id",
                "customer_unique_id",
                "customer_city",
                "customer_state"
            ]
        ],
        on="customer_id",
        how="inner",
        validate="many_to_one"
    )

    base = base.merge(
        itens_agg,
        on="order_id",
        how="left",
        validate="one_to_one"
    )

    base = base.merge(
        pagamentos_agg,
        on="order_id",
        how="left",
        validate="one_to_one"
    )

    base = base.merge(
        avaliacoes_agg,
        on="order_id",
        how="left",
        validate="one_to_one"
    )

    if base["order_id"].duplicated().any():
        raise ValueError(
            "Existem pedidos duplicados após a integração."
        )

    # Cálculo da recência

    base["recencia_pedido"] = (
        pd.Timestamp(REFERENCE_DATE)
        - base["order_purchase_timestamp"]
    ).dt.days

    # Consolidação por consumidor

    clientes_agg = base.groupby(
        "customer_unique_id",
        as_index=False
    ).agg(
        ultima_compra=(
            "order_purchase_timestamp", "max"
        ),
        frequencia=("order_id", "nunique"),
        valor_total=("valor_pedido", "sum"),
        total_itens=("total_itens", "sum"),
        valor_produtos_total=("valor_produtos", "sum"),
        frete_total=("valor_frete", "sum"),
        avaliacao_media=("avaliacao_media", "mean"),
        parcelas_media=("parcelas_media", "mean"),
        cidade=("customer_city", "last"),
        estado=("customer_state", "last"),
        recencia=("recencia_pedido", "min")
    )

    clientes_agg["ticket_medio"] = (
        clientes_agg["valor_total"]
        / clientes_agg["frequencia"]
    )

    clientes_agg["frete_medio"] = (
        clientes_agg["frete_total"]
        / clientes_agg["frequencia"]
    )

    clientes_agg["possui_avaliacao"] = (
        clientes_agg["avaliacao_media"]
        .notna()
        .astype(int)
    )

    clientes_agg.drop(
        columns=["frete_total"],
        inplace=True
    )

    # Organização final

    colunas = [
        "customer_unique_id",
        "ultima_compra",
        "frequencia",
        "valor_total",
        "total_itens",
        "valor_produtos_total",
        "frete_medio",
        "avaliacao_media",
        "parcelas_media",
        "cidade",
        "estado",
        "recencia",
        "ticket_medio",
        "possui_avaliacao"
    ]

    clientes_agg = clientes_agg[colunas]

    clientes_agg = clientes_agg.sort_values(
        "customer_unique_id"
    ).reset_index(drop=True)

    # Validações

    if clientes_agg["customer_unique_id"].isna().any():
        raise ValueError("Existem identificadores ausentes.")

    if clientes_agg["customer_unique_id"].duplicated().any():
        raise ValueError("Existem consumidores duplicados.")

    # Gravação temporária e validação

    arquivo_temp = INTEGRATED_FILE.with_name(
        INTEGRATED_FILE.stem + "_temporario.csv"
    )

    clientes_agg.to_csv(
        arquivo_temp,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n"
    )

    verificacao = pd.read_csv(
        arquivo_temp,
        dtype={"customer_unique_id": "string"}
    )

    if verificacao.shape != clientes_agg.shape:
        raise ValueError(
            "Falha na validação da gravação."
        )

    arquivo_temp.replace(INTEGRATED_FILE)

    print("\nIntegração concluída!")
    print("Pedidos entregues:", base["order_id"].nunique())
    print("Consumidores:", len(clientes_agg))
    print("Dimensões:", clientes_agg.shape)
    print("Arquivo:", INTEGRATED_FILE)


if __name__ == "__main__":
    integrar_dados()
