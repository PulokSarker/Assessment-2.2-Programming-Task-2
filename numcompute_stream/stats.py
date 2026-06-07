import numpy as np


class StreamingStats:

    def __init__(self, n_features=1, sliding_window=None):
        self.n_features = n_features
        self.sliding_window = sliding_window
        self._n = 0
        self._mean = np.zeros(n_features)
        self._M2 = np.zeros(n_features)   # sum of squared deviations (Welford)
        self._min = np.full(n_features, np.inf)
        self._max = np.full(n_features, -np.inf)
        # Rolling buffer for quantile / histogram estimation
        self._buffer = []

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update_stats(self, X_chunk):
        X = np.atleast_2d(np.array(X_chunk, dtype=float))
        if X.shape[0] == 1 and self.n_features > 1:
            # Allow (n_features,) input
            X = X.reshape(-1, self.n_features)

        # Drop NaN rows
        mask = ~np.any(np.isnan(X), axis=1)
        X = X[mask]
        if X.shape[0] == 0:
            return self

        # Welford batch update
        for xi in X:
            self._n += 1
            delta = xi - self._mean
            self._mean += delta / self._n
            delta2 = xi - self._mean
            self._M2 += delta * delta2

        self._min = np.minimum(self._min, X.min(axis=0))
        self._max = np.maximum(self._max, X.max(axis=0))

        # Rolling buffer
        if self.sliding_window is not None:
            self._buffer.extend(X.tolist())
            if len(self._buffer) > self.sliding_window:
                self._buffer = self._buffer[-self.sliding_window:]
        else:
            self._buffer.extend(X.tolist())
        return self

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def mean(self):
        """Return running mean per feature."""
        return self._mean.copy()

    def variance(self, ddof=1):
        """Return running variance per feature (population if ddof=0)."""
        if self._n < 2:
            return np.zeros(self.n_features)
        denom = self._n - ddof if ddof else self._n
        return self._M2 / max(denom, 1)

    def std(self, ddof=1):
        """Return running standard deviation per feature."""
        return np.sqrt(self.variance(ddof))

    def min(self):
        return self._min.copy()

    def max(self):
        return self._max.copy()

    def count(self):
        return self._n

    def quantiles(self, q=(0.25, 0.5, 0.75)):
        if not self._buffer:
            return {qi: np.zeros(self.n_features) for qi in q}
        arr = np.array(self._buffer)
        return {qi: np.nanpercentile(arr, qi * 100, axis=0) for qi in q}

    def histogram(self, feature_idx=0, bins=20):
        if not self._buffer:
            return np.array([]), np.array([])
        arr = np.array(self._buffer)[:, feature_idx]
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return np.array([]), np.array([])
        counts, edges = np.histogram(arr, bins=bins)
        return counts, edges

    def reset(self):
        self.__init__(self.n_features, self.sliding_window)
        return self

    def summary(self):
        return {
            "n": self._n,
            "mean": self.mean(),
            "std": self.std(),
            "min": self.min(),
            "max": self.max(),
        }


# ------------------------------------------------------------------
# Module-level convenience functions
# ------------------------------------------------------------------

def chunk_mean(X_chunk):
    """Mean per column, ignoring NaNs."""
    return np.nanmean(np.atleast_2d(np.array(X_chunk, dtype=float)), axis=0)


def chunk_variance(X_chunk, ddof=1):
    """Variance per column, ignoring NaNs."""
    return np.nanvar(np.atleast_2d(np.array(X_chunk, dtype=float)), axis=0, ddof=ddof)


def chunk_quantiles(X_chunk, q=(0.25, 0.5, 0.75)):
    arr = np.atleast_2d(np.array(X_chunk, dtype=float))
    return {qi: np.nanpercentile(arr, qi * 100, axis=0) for qi in q}


def chunk_histogram(X_chunk, feature_idx=0, bins=20):
    arr = np.atleast_2d(np.array(X_chunk, dtype=float))[:, feature_idx]
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.array([]), np.array([])
    return np.histogram(arr, bins=bins)