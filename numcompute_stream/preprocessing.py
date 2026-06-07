import numpy as np


class StandardScaler:
    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self._n = 0
        self._mean = None
        self._M2 = None

    def partial_fit(self, X):
        X = self._validate(X)
        n_features = X.shape[1]
        if self._mean is None:
            self._mean = np.zeros(n_features)
            self._M2 = np.zeros(n_features)

        for xi in X:
            # Skip rows with NaN
            if np.any(np.isnan(xi)):
                continue
            self._n += 1
            delta = xi - self._mean
            self._mean += delta / self._n
            delta2 = xi - self._mean
            self._M2 += delta * delta2
        return self

    def transform(self, X):
        X = self._validate(X).copy()
        if self._mean is None:
            raise RuntimeError("Call partial_fit before transform.")
        if self.with_mean:
            X -= self._mean
        if self.with_std:
            std = self._std()
            std[std == 0] = 1.0   # avoid division by zero (zero-variance feature)
            X /= std
        return X

    def fit_transform(self, X):
        return self.partial_fit(X).transform(X)

    def inverse_transform(self, X):
        X = self._validate(X).copy()
        if self.with_std:
            X *= self._std()
        if self.with_mean:
            X += self._mean
        return X

    def _std(self):
        if self._n < 2:
            return np.ones_like(self._mean)
        return np.sqrt(self._M2 / (self._n - 1))

    @property
    def mean_(self):
        return self._mean

    @property
    def scale_(self):
        return self._std()

    @staticmethod
    def _validate(X):
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X


class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self._data_min = None
        self._data_max = None

    def partial_fit(self, X):
        X = self._validate(X)
        chunk_min = np.nanmin(X, axis=0)
        chunk_max = np.nanmax(X, axis=0)
        if self._data_min is None:
            self._data_min = chunk_min
            self._data_max = chunk_max
        else:
            self._data_min = np.minimum(self._data_min, chunk_min)
            self._data_max = np.maximum(self._data_max, chunk_max)
        return self

    def transform(self, X):
        X = self._validate(X).copy()
        if self._data_min is None:
            raise RuntimeError("Call partial_fit before transform.")
        rng = self._data_max - self._data_min
        rng[rng == 0] = 1.0
        lo, hi = self.feature_range
        X = (X - self._data_min) / rng * (hi - lo) + lo
        return X

    def fit_transform(self, X):
        return self.partial_fit(X).transform(X)

    @staticmethod
    def _validate(X):
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X


class Imputer:
    def __init__(self, strategy="mean"):
        if strategy not in ("mean", "median"):
            raise ValueError("strategy must be 'mean' or 'median'")
        self.strategy = strategy
        # For mean: running Welford
        self._n = None
        self._mean = None
        self._M2 = None
        # For median: buffer approach
        self._buffer = None

    def partial_fit(self, X):
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_features = X.shape[1]

        if self._n is None:
            self._n = np.zeros(n_features)
            self._mean = np.zeros(n_features)
            self._M2 = np.zeros(n_features)
            if self.strategy == "median":
                self._buffer = [[] for _ in range(n_features)]

        for j in range(n_features):
            col = X[:, j]
            valid = col[~np.isnan(col)]
            for v in valid:
                self._n[j] += 1
                delta = v - self._mean[j]
                self._mean[j] += delta / self._n[j]
                self._M2[j] += delta * (v - self._mean[j])
            if self.strategy == "median":
                self._buffer[j].extend(valid.tolist())
        return self

    def transform(self, X):
        X = np.array(X, dtype=float).copy()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self._mean is None:
            raise RuntimeError("Call partial_fit before transform.")

        if self.strategy == "mean":
            fill = self._mean
        else:
            fill = np.array([
                np.median(b) if b else 0.0
                for b in self._buffer
            ])

        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = fill[j]
        return X

    def fit_transform(self, X):
        return self.partial_fit(X).transform(X)


class OneHotEncoder:
    def __init__(self, handle_unknown="ignore"):
        self.handle_unknown = handle_unknown
        self.categories_ = None  # list of arrays, one per feature

    def partial_fit(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_features = X.shape[1]

        if self.categories_ is None:
            self.categories_ = [set() for _ in range(n_features)]

        for j in range(n_features):
            self.categories_[j].update(X[:, j].tolist())
        return self

    def transform(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.categories_ is None:
            raise RuntimeError("Call partial_fit before transform.")

        parts = []
        for j, cats in enumerate(self.categories_):
            sorted_cats = sorted(cats)
            idx_map = {c: i for i, c in enumerate(sorted_cats)}
            col = X[:, j]
            ohe = np.zeros((len(col), len(sorted_cats)), dtype=float)
            for i, v in enumerate(col):
                if v in idx_map:
                    ohe[i, idx_map[v]] = 1.0
                elif self.handle_unknown == "error":
                    raise ValueError(f"Unknown category: {v!r}")
                # else ignore → row stays zeros
            parts.append(ohe)
        return np.hstack(parts) if parts else np.zeros((X.shape[0], 0))

    def fit_transform(self, X):
        return self.partial_fit(X).transform(X)