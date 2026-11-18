from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas de fenêtre : on écrit juste des fichiers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

THRESHOLD = 0.5
CLASS_ORDER = [0, 1]  # 0 = No churn, 1 = Churn (Yes)


def positive_scores(pipeline, X) -> np.ndarray:
    """Score continu de la classe 1, colonne trouvée via classes_ (pipeline déjà fitté)."""
    idx = list(pipeline.classes_).index(1)
    return pipeline.predict_proba(X)[:, idx]


def evaluate(pipeline, X, y, threshold: float = THRESHOLD) -> dict:
    """AP sur scores continus ; precision/recall/F1 au seuil fixé."""
    scores = positive_scores(pipeline, X)
    pred = (scores >= threshold).astype(int)
    return {
        "val_average_precision": float(average_precision_score(y, scores)),
        "val_precision": float(precision_score(y, pred, zero_division=0)),
        "val_recall": float(recall_score(y, pred, zero_division=0)),
        "val_f1": float(f1_score(y, pred, zero_division=0)),
    }


def confusion(pipeline, X, y, threshold: float = THRESHOLD) -> np.ndarray:
    """Lignes = classe réelle, colonnes = classe prédite, ordre [0, 1]."""
    pred = (positive_scores(pipeline, X) >= threshold).astype(int)
    return confusion_matrix(y, pred, labels=CLASS_ORDER)


def save_confusion_matrix(pipeline, X, y, path: Path, threshold: float = THRESHOLD) -> Path:
    """Enregistre la matrice de confusion en PNG (artefact pour MLflow)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion(pipeline, X, y, threshold)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No (0)", "Yes (1)"])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Validation, seuil = {threshold}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path