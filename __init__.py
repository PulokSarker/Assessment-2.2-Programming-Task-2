from .stats import StreamingStats, chunk_mean, chunk_variance
from .metrics import (
    StreamingAccuracy, StreamingConfusionMatrix,
    StreamingPrecisionRecallF1, StreamingAUC,
    accuracy_score, precision_recall_f1, confusion_matrix,
)
from .preprocessing import StandardScaler, MinMaxScaler, Imputer, OneHotEncoder
from .tree import DecisionTreeClassifier
from .ensemble import RandomForestClassifier, BaggingClassifier, EnsembleClassifier
from .pipeline import Pipeline
from .stream import StreamTrainer
from .io import load_csv, save_csv, stream_csv, generate_synthetic_dataset
from . import visualise

__version__ = "1.0.0"
__all__ = [
    "StreamingStats", "chunk_mean", "chunk_variance",
    "StreamingAccuracy", "StreamingConfusionMatrix",
    "StreamingPrecisionRecallF1", "StreamingAUC",
    "accuracy_score", "precision_recall_f1", "confusion_matrix",
    "StandardScaler", "MinMaxScaler", "Imputer", "OneHotEncoder",
    "DecisionTreeClassifier",
    "RandomForestClassifier", "BaggingClassifier", "EnsembleClassifier",
    "Pipeline",
    "StreamTrainer",
    "load_csv", "save_csv", "stream_csv", "generate_synthetic_dataset",
    "visualise",
]