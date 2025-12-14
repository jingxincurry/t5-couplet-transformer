from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'
PROCESSED_DIR = ROOT_DIR / 'data' / 'processed'
LOGS_DIR = ROOT_DIR / 'logs'
MODELS_DIR = ROOT_DIR / 'models'
PRETERAINED_MODELS_DIR = ROOT_DIR / 'pretrained'

SEQ_LEN = 32
BATCH_SIZE = 16
EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
LEARNING_RATE = 1e-5  # ！！！注意调整学习率
EPOCHS = 30