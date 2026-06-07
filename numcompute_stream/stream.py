import time
import sys
import numpy as np
from .metrics import StreamingAccuracy


class StreamTrainer:
    def __init__(self, model, scaler=None, classes=None, log_memory=True):
        self.model = model
        self.scaler = scaler
        self.classes = classes
        self.log_memory = log_memory

        self._acc_tracker = StreamingAccuracy()
        self.chunk_logs_ = []
        self._chunk_idx = 0

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def fit_chunk(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)
        n_samples = len(X)

        # Scale if scaler provided
        if self.scaler is not None:
            self.scaler.partial_fit(X)
            Xt = self.scaler.transform(X)
        else:
            Xt = X

        # Time the fit
        t0 = time.perf_counter()
        if self.classes is not None and hasattr(self.model, "partial_fit"):
            try:
                self.model.partial_fit(Xt, y, classes=self.classes)
            except TypeError:
                self.model.partial_fit(Xt, y)
        elif hasattr(self.model, "partial_fit"):
            self.model.partial_fit(Xt, y)
        else:
            self.model.fit(Xt, y)
        fit_time = time.perf_counter() - t0

        # Score on the training chunk (online accuracy)
        y_pred = self.model.predict(Xt)
        chunk_acc = float(np.mean(y_pred == y))
        self._acc_tracker.update(y, y_pred)
        cum_acc = self._acc_tracker.result()

        log_entry = {
            "chunk": self._chunk_idx,
            "n_samples": n_samples,
            "accuracy": chunk_acc,
            "cumulative_accuracy": cum_acc,
            "fit_time_s": fit_time,
        }

        if self.log_memory:
            log_entry["memory_bytes"] = self._estimate_memory()

        self.chunk_logs_.append(log_entry)
        self._chunk_idx += 1
        return self

    def score_chunk(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        y_pred = self.model.predict(X)
        return float(np.mean(y_pred == y))

    def predict(self, X):
        """Predict labels for X using the current model."""
        X = np.array(X, dtype=float)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict(X)

    # ------------------------------------------------------------------
    # Streaming helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_chunks(X, y, n_chunks=10, shuffle=False, random_state=None):
        X = np.array(X, dtype=float)
        y = np.array(y)
        n = len(X)
        if shuffle:
            rng = np.random.default_rng(random_state)
            idx = rng.permutation(n)
            X, y = X[idx], y[idx]
        chunk_size = max(1, n // n_chunks)
        for i in range(0, n, chunk_size):
            yield X[i:i + chunk_size], y[i:i + chunk_size]

    def run_stream(self, X, y, n_chunks=10, shuffle=False, random_state=None):
        for X_chunk, y_chunk in self.make_chunks(X, y, n_chunks, shuffle, random_state):
            self.fit_chunk(X_chunk, y_chunk)
        return self

    # ------------------------------------------------------------------
    # Logging / memory
    # ------------------------------------------------------------------

    def _estimate_memory(self):
        """Rough estimate of model size in bytes via sys.getsizeof recursion."""
        try:
            return sys.getsizeof(self.model)
        except Exception:
            return 0

    def get_metric_history(self, metric="accuracy"):
        return [log[metric] for log in self.chunk_logs_ if metric in log]

    def summary(self):
        """Print a formatted summary of per-chunk logs."""
        print(f"{'Chunk':>6} {'Samples':>8} {'Acc':>8} {'CumAcc':>8} {'Time(s)':>10}")
        print("-" * 45)
        for log in self.chunk_logs_:
            print(
                f"{log['chunk']:>6} "
                f"{log['n_samples']:>8} "
                f"{log['accuracy']:>8.4f} "
                f"{log['cumulative_accuracy']:>8.4f} "
                f"{log['fit_time_s']:>10.4f}"
            )