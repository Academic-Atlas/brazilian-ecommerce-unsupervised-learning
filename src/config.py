
from pathlib import Path

# Diretório principal
ROOT_DIR = Path(__file__).resolve().parents[1]

# Diretórios
DATA_ORIGINAL = ROOT_DIR / "dados" / "original"
DATA_PROCESSED = ROOT_DIR / "dados" / "processados"
DATA_PREPARED = ROOT_DIR / "dados" / "preparados"

MODELS_DIR = ROOT_DIR / "modelos"
REPORTS_DIR = ROOT_DIR / "relatorios"

# Arquivos originais
CUSTOMERS_FILE = DATA_ORIGINAL / "olist_customers_dataset.csv"
ORDERS_FILE = DATA_ORIGINAL / "olist_orders_dataset.csv"
ITEMS_FILE = DATA_ORIGINAL / "olist_order_items_dataset.csv"
PAYMENTS_FILE = DATA_ORIGINAL / "olist_order_payments_dataset.csv"
REVIEWS_FILE = DATA_ORIGINAL / "olist_order_reviews_dataset.csv"

# Arquivos processados
INTEGRATED_FILE = DATA_PROCESSED / "clientes_2017_integrado.csv"
CLEAN_FILE = DATA_PROCESSED / "clientes_2017_tratado.csv"

# Arquivos preparados
TRAIN_FILE = DATA_PREPARED / "X_treino.csv"
TEST_FILE = DATA_PREPARED / "X_teste.csv"
TEST_IDS_FILE = DATA_PREPARED / "ids_teste.csv"

# Modelos
PREPROCESSOR_FILE = MODELS_DIR / "preprocessador.joblib"
KMEANS_FILE = MODELS_DIR / "kmeans.joblib"
DBSCAN_FILE = MODELS_DIR / "dbscan.joblib"
DBSCAN_CORE_FILE = MODELS_DIR / "dbscan_core.joblib"

# Relatórios
METRICS_FILE = REPORTS_DIR / "metricas_modelos.csv"
PREDICTIONS_FILE = REPORTS_DIR / "predicoes_clientes.csv"

# Período de análise
START_DATE = "2017-01-01"
END_DATE = "2018-01-01"

# Data de referência para recência
REFERENCE_DATE = "2018-01-01"

# Identificador
ID_COLUMN = "customer_unique_id"

# Variáveis utilizadas nos modelos
FEATURES = [
    "recencia",
    "frequencia",
    "valor_total",
    "ticket_medio",
    "total_itens",
    "valor_produtos_total",
    "frete_medio",
    "avaliacao_media",
    "possui_avaliacao",
    "parcelas_media"
]

# Parâmetros gerais
RANDOM_STATE = 42
TEST_SIZE = 0.20

# K-Means
K_MIN = 2
K_MAX = 8

# DBSCAN
DBSCAN_EPS = 1.5
DBSCAN_MIN_SAMPLES = 10
DBSCAN_SAMPLE_SIZE = 15000

# Avaliação
SILHOUETTE_SAMPLE_SIZE = 5000

# Criar diretórios
def create_directories():

    diretorios = [
        DATA_ORIGINAL,
        DATA_PROCESSED,
        DATA_PREPARED,
        MODELS_DIR,
        REPORTS_DIR
    ]

    for diretorio in diretorios:
        diretorio.mkdir(parents=True, exist_ok=True)

