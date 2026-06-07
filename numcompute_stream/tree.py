import numpy as np


# ---------------------------------------------------------------------------
# Impurity functions (vectorised)
# ---------------------------------------------------------------------------

def _gini(y):
    """Gini impurity of label array y."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(1.0 - np.dot(p, p))


def _entropy(y):
    """Entropy of label array y."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    # Clip to avoid log(0)
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.dot(p, np.log2(p)))


_CRITERION = {"gini": _gini, "entropy": _entropy}


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = (
        "feature", "threshold", "left", "right",
        "is_leaf", "class_counts", "prediction"
    )

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.class_counts = None
        self.prediction = None


# ---------------------------------------------------------------------------
# DecisionTreeClassifier
# ---------------------------------------------------------------------------

class DecisionTreeClassifier:
    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        criterion="gini",
        max_features=None,
        random_state=None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        if criterion not in _CRITERION:
            raise ValueError(f"criterion must be 'gini' or 'entropy', got {criterion!r}")
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

        self._impurity = _CRITERION[criterion]
        self._rng = np.random.default_rng(random_state)
        self._root = None
        self.classes_ = None
        self.n_features_ = None

        # Streaming buffer: accumulate all seen data
        self._X_buf = None
        self._y_buf = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """Full fit on a dataset (clears buffer)."""
        X, y = self._validate(X, y)
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        self._X_buf = X.copy()
        self._y_buf = y.copy()
        self._root = self._build(X, y, depth=0)
        return self

    def partial_fit(self, X_chunk, y_chunk, classes=None):
        X_chunk, y_chunk = self._validate(X_chunk, y_chunk)

        if classes is not None:
            if self.classes_ is None:
                self.classes_ = np.array(sorted(classes))
            else:
                self.classes_ = np.array(sorted(set(self.classes_.tolist()) | set(classes)))

        if self._X_buf is None:
            self._X_buf = X_chunk.copy()
            self._y_buf = y_chunk.copy()
        else:
            self._X_buf = np.vstack([self._X_buf, X_chunk])
            self._y_buf = np.concatenate([self._y_buf, y_chunk])

        # Update known classes
        seen = np.unique(self._y_buf)
        if self.classes_ is None:
            self.classes_ = seen
        else:
            self.classes_ = np.array(sorted(set(self.classes_.tolist()) | set(seen.tolist())))

        self.n_features_ = self._X_buf.shape[1]
        self._root = self._build(self._X_buf, self._y_buf, depth=0)
        return self

    def predict(self, X):
        """Predict class labels for X."""
        X, _ = self._validate(X)
        if self._root is None:
            raise RuntimeError("Tree is not fitted. Call fit or partial_fit first.")
        return np.array([self._predict_row(x, self._root) for x in X])

    def predict_proba(self, X):
        """Predict class probabilities for X."""
        X, _ = self._validate(X)
        if self._root is None:
            raise RuntimeError("Tree is not fitted.")
        n_classes = len(self.classes_)
        proba = np.zeros((len(X), n_classes))
        for i, x in enumerate(X):
            counts = self._predict_proba_row(x, self._root)
            total = counts.sum()
            proba[i] = counts / max(total, 1)
        return proba

    def score(self, X, y):
        X, y = self._validate(X, y)
        return float(np.mean(self.predict(X) == y))

    # ------------------------------------------------------------------
    # Internal build
    # ------------------------------------------------------------------

    def _build(self, X, y, depth):
        node = _Node()
        node.class_counts = self._class_counts(y)
        node.prediction = self.classes_[np.argmax(node.class_counts)]

        # Stopping criteria
        pure = len(np.unique(y)) == 1
        too_small = len(y) < self.min_samples_split
        too_deep = self.max_depth is not None and depth >= self.max_depth

        if pure or too_small or too_deep:
            node.is_leaf = True
            return node

        feature, threshold = self._best_split(X, y)
        if feature is None:
            node.is_leaf = True
            return node

        node.feature = feature
        node.threshold = threshold

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        # Guard: avoid empty splits
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            node.is_leaf = True
            return node

        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[right_mask], y[right_mask], depth + 1)
        return node

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        features = self._select_features(n_features)

        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        parent_impurity = self._impurity(y)

        for f in features:
            col = X[:, f]
            # Ignore NaN
            valid = ~np.isnan(col)
            if valid.sum() < 2:
                continue
            vals = np.unique(col[valid])
            if len(vals) < 2:
                continue
            # Candidate thresholds: midpoints
            thresholds = (vals[:-1] + vals[1:]) / 2.0

            # Vectorised gain computation
            for t in thresholds:
                left_mask = col <= t
                right_mask = ~left_mask
                nl, nr = left_mask.sum(), right_mask.sum()
                if nl == 0 or nr == 0:
                    continue
                gain = parent_impurity - (
                    nl / n_samples * self._impurity(y[left_mask]) +
                    nr / n_samples * self._impurity(y[right_mask])
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feature = f
                    best_threshold = t

        return best_feature, best_threshold

    def _select_features(self, n_features):
        mf = self.max_features
        if mf is None:
            return np.arange(n_features)
        if mf == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif mf == "log2":
            k = max(1, int(np.log2(n_features)))
        elif isinstance(mf, float):
            k = max(1, int(mf * n_features))
        elif isinstance(mf, int):
            k = min(mf, n_features)
        else:
            return np.arange(n_features)
        return self._rng.choice(n_features, size=k, replace=False)

    def _class_counts(self, y):
        counts = np.zeros(len(self.classes_) if self.classes_ is not None else 1)
        if self.classes_ is None:
            return counts
        for i, c in enumerate(self.classes_):
            counts[i] = np.sum(y == c)
        return counts

    def _predict_row(self, x, node):
        if node.is_leaf:
            return node.prediction
        if np.isnan(x[node.feature]) or x[node.feature] <= node.threshold:
            return self._predict_row(x, node.left)
        return self._predict_row(x, node.right)

    def _predict_proba_row(self, x, node):
        if node.is_leaf:
            return node.class_counts
        if np.isnan(x[node.feature]) or x[node.feature] <= node.threshold:
            return self._predict_proba_row(x, node.left)
        return self._predict_proba_row(x, node.right)

    @staticmethod
    def _validate(X, y=None):
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y is not None:
            y = np.array(y)
            if len(y) != len(X):
                raise ValueError(
                    f"X and y have different lengths: {len(X)} vs {len(y)}"
                )
            return X, y
        return X, None