import numpy as np
from .tree import DecisionTreeClassifier


# ---------------------------------------------------------------------------
# Generic Ensemble
# ---------------------------------------------------------------------------

class EnsembleClassifier:
    def __init__(self, estimators, voting="hard"):
        self.estimators = estimators
        self.voting = voting
        self.classes_ = None

    def partial_fit(self, X_chunk, y_chunk, classes=None):
        X_chunk = np.array(X_chunk, dtype=float)
        y_chunk = np.array(y_chunk)

        if classes is not None:
            self.classes_ = np.array(sorted(classes))
        elif self.classes_ is None:
            self.classes_ = np.array(sorted(np.unique(y_chunk).tolist()))
        else:
            new = set(np.unique(y_chunk).tolist())
            self.classes_ = np.array(sorted(set(self.classes_.tolist()) | new))

        for est in self.estimators:
            est.partial_fit(X_chunk, y_chunk, classes=self.classes_)
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        if self.voting == "soft":
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]
        # Hard voting: collect predictions from each estimator
        preds = np.array([est.predict(X) for est in self.estimators])
        # Majority vote per sample
        result = []
        for col in preds.T:
            vals, counts = np.unique(col, return_counts=True)
            result.append(vals[np.argmax(counts)])
        return np.array(result)

    def predict_proba(self, X):
        X = np.array(X, dtype=float)
        probas = [est.predict_proba(X) for est in self.estimators]
        return np.mean(probas, axis=0)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.array(y)))


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

class RandomForestClassifier:
    def __init__(
        self,
        n_estimators=10,
        max_depth=5,
        min_samples_split=2,
        criterion="gini",
        max_features="sqrt",
        bootstrap=True,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self._rng = np.random.default_rng(random_state)
        self.estimators_ = [
            DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                criterion=criterion,
                max_features=max_features,
                random_state=int(self._rng.integers(0, 2**31)),
            )
            for _ in range(n_estimators)
        ]
        self.classes_ = None

    def partial_fit(self, X_chunk, y_chunk, classes=None):
        X_chunk = np.array(X_chunk, dtype=float)
        y_chunk = np.array(y_chunk)
        n_samples = len(X_chunk)

        if classes is not None:
            self.classes_ = np.array(sorted(classes))
        elif self.classes_ is None:
            self.classes_ = np.array(sorted(np.unique(y_chunk).tolist()))
        else:
            new = set(np.unique(y_chunk).tolist())
            self.classes_ = np.array(sorted(set(self.classes_.tolist()) | new))

        for tree in self.estimators_:
            if self.bootstrap:
                idx = self._rng.integers(0, n_samples, size=n_samples)
                X_b, y_b = X_chunk[idx], y_chunk[idx]
            else:
                X_b, y_b = X_chunk, y_chunk
            tree.partial_fit(X_b, y_b, classes=self.classes_)
        return self

    def fit(self, X, y):
        """Full fit (clears existing trees)."""
        self._rng = np.random.default_rng(self.random_state)
        self.estimators_ = [
            DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                max_features=self.max_features,
                random_state=int(self._rng.integers(0, 2**31)),
            )
            for _ in range(self.n_estimators)
        ]
        self.classes_ = None
        return self.partial_fit(X, y)

    def predict(self, X):
        X = np.array(X, dtype=float)
        preds = np.array([tree.predict(X) for tree in self.estimators_])
        result = []
        for col in preds.T:
            vals, counts = np.unique(col, return_counts=True)
            result.append(vals[np.argmax(counts)])
        return np.array(result)

    def predict_proba(self, X):
        X = np.array(X, dtype=float)
        probas = [tree.predict_proba(X) for tree in self.estimators_]
        return np.mean(probas, axis=0)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.array(y)))


# ---------------------------------------------------------------------------
# Bagging Classifier (generic)
# ---------------------------------------------------------------------------

class BaggingClassifier:
    def __init__(
        self,
        base_estimator=None,
        n_estimators=10,
        max_samples=1.0,
        random_state=None,
    ):
        if base_estimator is None:
            base_estimator = DecisionTreeClassifier()
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

        self._rng = np.random.default_rng(random_state)
        self.estimators_ = []
        self.classes_ = None

    def partial_fit(self, X_chunk, y_chunk, classes=None):
        X_chunk = np.array(X_chunk, dtype=float)
        y_chunk = np.array(y_chunk)
        n_samples = len(X_chunk)
        k = max(1, int(n_samples * self.max_samples))

        if classes is not None:
            self.classes_ = np.array(sorted(classes))
        elif self.classes_ is None:
            self.classes_ = np.array(sorted(np.unique(y_chunk).tolist()))

        # Grow ensemble if needed (first call)
        while len(self.estimators_) < self.n_estimators:
            import copy
            self.estimators_.append(copy.deepcopy(self.base_estimator))

        for est in self.estimators_:
            idx = self._rng.integers(0, n_samples, size=k)
            est.partial_fit(X_chunk[idx], y_chunk[idx], classes=self.classes_)
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        preds = np.array([est.predict(X) for est in self.estimators_])
        result = []
        for col in preds.T:
            vals, counts = np.unique(col, return_counts=True)
            result.append(vals[np.argmax(counts)])
        return np.array(result)

    def predict_proba(self, X):
        X = np.array(X, dtype=float)
        probas = [est.predict_proba(X) for est in self.estimators_]
        return np.mean(probas, axis=0)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.array(y)))