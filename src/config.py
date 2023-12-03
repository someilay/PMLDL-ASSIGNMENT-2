# Config file
from pathlib import Path


class ModelTypes:
    WITHOUT = 'without-feature'
    WITH = 'with-feature'


class ModelMetrics:
    RECALL = 'recall'
    PRECISION = 'precision'


MODEL2RECALL_SAVE_PATH = {
    ModelTypes.WITHOUT: 'best-recall-model-without-features.pt',
    ModelTypes.WITH: 'best-recall-model-with-features.pt',
}
MODEL2PRECISION_SAVE_PATH = {
    ModelTypes.WITHOUT: 'best-precision-model-without-features.pt',
    ModelTypes.WITH: 'best-precision-model-with-features.pt',
}
MODEL2HISTORY = {
    ModelTypes.WITHOUT: 'history-without',
    ModelTypes.WITH: 'history-with',
}


BEST_RECALL_MODEL_PATH = Path('models')
BEST_PRECISION_MODEL_PATH = Path('models')
METRICS_HISTORY_PATH = Path('src')
