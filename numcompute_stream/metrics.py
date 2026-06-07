import numpy as np


class StreamingMetric:
    """Base class for streaming metrics."""

    def update(self, y_true, y_pred):
        raise NotImplementedError

    def result(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


# -------------------------------------------------------------------------
# Accuracy
# -------------------------------------------------------------------------

class StreamingAccuracy(StreamingMetric):

    def __init__(self, rolling_window=None):
        self.rolling_window = rolling_window
        self._total = 0
        self._correct = 0
        self._history_true = []
        self._history_pred = []

    def update(self, y_true, y_pred):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        if len(y_true) != len(y_pred):
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        self._total += len(y_true)
        self._correct += int(np.sum(y_true == y_pred))
        if self.rolling_window is not None:
            self._history_true.extend(y_true.tolist())
            self._history_pred.extend(y_pred.tolist())
            self._history_true = self._history_true[-self.rolling_window:]
            self._history_pred = self._history_pred[-self.rolling_window:]
        return self

    def result(self):
        if self.rolling_window is not None:
            if not self._history_true:
                return 0.0
            yt = np.array(self._history_true)
            yp = np.array(self._history_pred)
            return float(np.mean(yt == yp))
        if self._total == 0:
            return 0.0
        return self._correct / self._total

    def reset(self):
        self._total = 0
        self._correct = 0
        self._history_true = []
        self._history_pred = []
        return self


# -------------------------------------------------------------------------
# Confusion Matrix
# -------------------------------------------------------------------------

class StreamingConfusionMatrix(StreamingMetric):

    def __init__(self, classes=None):
        self._classes = list(classes) if classes is not None else []
        self._matrix = {}

    def update(self, y_true, y_pred):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        for yt, yp in zip(y_true, y_pred):
            yt, yp = int(yt), int(yp)
            if yt not in self._classes:
                self._classes.append(yt)
            if yp not in self._classes:
                self._classes.append(yp)
            self._matrix[(yt, yp)] = self._matrix.get((yt, yp), 0) + 1
        return self

    def result(self):
        """Return confusion matrix as a 2-D NumPy array."""
        classes = sorted(self._classes)
        n = len(classes)
        mat = np.zeros((n, n), dtype=int)
        idx = {c: i for i, c in enumerate(classes)}
        for (yt, yp), cnt in self._matrix.items():
            if yt in idx and yp in idx:
                mat[idx[yt], idx[yp]] += cnt
        return mat, classes

    def reset(self):
        self._classes = []
        self._matrix = {}
        return self


# -------------------------------------------------------------------------
# Precision / Recall / F1
# -------------------------------------------------------------------------

class StreamingPrecisionRecallF1(StreamingMetric):

    def __init__(self, classes=None):
        self._cm = StreamingConfusionMatrix(classes)

    def update(self, y_true, y_pred):
        self._cm.update(y_true, y_pred)
        return self

    def result(self):
        mat, classes = self._cm.result()
        n = len(classes)
        precision = np.zeros(n)
        recall = np.zeros(n)
        f1 = np.zeros(n)
        for i in range(n):
            tp = mat[i, i]
            fp = mat[:, i].sum() - tp
            fn = mat[i, :].sum() - tp
            precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            denom = precision[i] + recall[i]
            f1[i] = 2 * precision[i] * recall[i] / denom if denom > 0 else 0.0
        return {
            "classes": classes,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "macro_precision": float(precision.mean()),
            "macro_recall": float(recall.mean()),
            "macro_f1": float(f1.mean()),
        }

    def reset(self):
        self._cm.reset()
        return self


# -------------------------------------------------------------------------
# AUC (binary, streaming via trapezoidal approximation)
# -------------------------------------------------------------------------

class StreamingAUC(StreamingMetric):
    def __init__(self, rolling_window=None):
        self.rolling_window = rolling_window
        self._y_true = []
        self._y_score = []

    def update(self, y_true, y_score):
        y_true = np.asarray(y_true).ravel().tolist()
        y_score = np.asarray(y_score).ravel().tolist()
        self._y_true.extend(y_true)
        self._y_score.extend(y_score)
        if self.rolling_window is not None:
            self._y_true = self._y_true[-self.rolling_window:]
            self._y_score = self._y_score[-self.rolling_window:]
        return self

    def result(self):
        if len(set(self._y_true)) < 2:
            return 0.5  # undefined – return chance level
        yt = np.array(self._y_true)
        ys = np.array(self._y_score)
        # Sort by descending score
        order = np.argsort(-ys)
        yt_sorted = yt[order]
        pos = yt_sorted == 1
        neg = yt_sorted == 0
        tpr = np.cumsum(pos) / max(pos.sum(), 1)
        fpr = np.cumsum(neg) / max(neg.sum(), 1)
        # Prepend (0,0)
        tpr = np.concatenate([[0], tpr])
        fpr = np.concatenate([[0], fpr])
        trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
        return float(trapz(tpr, fpr))

    def reset(self):
        self._y_true = []
        self._y_score = []
        return self


# -------------------------------------------------------------------------
# Convenience top-level functions
# -------------------------------------------------------------------------

def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(y_true == y_pred))


def precision_recall_f1(y_true, y_pred, average="macro"):
    m = StreamingPrecisionRecallF1()
    m.update(y_true, y_pred)
    res = m.result()
    if average == "macro":
        return res["macro_precision"], res["macro_recall"], res["macro_f1"]
    return res["precision"], res["recall"], res["f1"]


def confusion_matrix(y_true, y_pred):
    cm = StreamingConfusionMatrix()
    cm.update(y_true, y_pred)
    return cm.result()