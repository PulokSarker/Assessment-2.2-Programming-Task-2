import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import (
    RandomForestClassifier, BaggingClassifier, EnsembleClassifier
)


# ---------------------------------------------------------------------------
# Helper datasets
# ---------------------------------------------------------------------------

def make_linearly_separable(n=100, seed=0):
    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal((n // 2, 2)) + np.array([-2, -2])
    X1 = rng.standard_normal((n // 2, 2)) + np.array([2, 2])
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


def make_multiclass(n=150, seed=1):
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for cls, center in enumerate([(-3, 0), (0, 3), (3, 0)]):
        Xs.append(rng.standard_normal((n // 3, 2)) + np.array(center))
        ys.extend([cls] * (n // 3))
    return np.vstack(Xs), np.array(ys)


# ---------------------------------------------------------------------------
# DecisionTreeClassifier
# ---------------------------------------------------------------------------

class TestDecisionTreeClassifier:

    def test_fit_predict_binary(self):
        X, y = make_linearly_separable()
        tree = DecisionTreeClassifier(max_depth=3, random_state=0)
        tree.fit(X, y)
        acc = (tree.predict(X) == y).mean()
        assert acc > 0.90, f"Expected >90% accuracy, got {acc:.2f}"

    def test_partial_fit_streaming(self):
        X, y = make_linearly_separable(n=200)
        tree = DecisionTreeClassifier(max_depth=4, random_state=0)
        chunks = np.array_split(np.arange(200), 5)
        for idx in chunks:
            tree.partial_fit(X[idx], y[idx])
        acc = (tree.predict(X) == y).mean()
        assert acc > 0.85

    def test_partial_fit_accumulates_data(self):
        X, y = make_linearly_separable(n=60)
        tree = DecisionTreeClassifier(max_depth=3, random_state=42)
        tree.partial_fit(X[:30], y[:30])
        tree.partial_fit(X[30:], y[30:])
        # After two chunks, buffer should have all 60 samples
        assert len(tree._X_buf) == 60

    def test_predict_proba_sums_to_one(self):
        X, y = make_linearly_separable()
        tree = DecisionTreeClassifier(random_state=0)
        tree.fit(X, y)
        proba = tree.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-10)

    def test_max_depth_limits_tree(self):
        X, y = make_multiclass()
        tree = DecisionTreeClassifier(max_depth=1, random_state=0)
        tree.fit(X, y)
        # With depth=1, predictions are limited but should still be valid
        preds = tree.predict(X)
        assert set(preds).issubset(set(y))

    def test_gini_vs_entropy(self):
        X, y = make_linearly_separable()
        t_gini = DecisionTreeClassifier(criterion="gini", random_state=0).fit(X, y)
        t_ent = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(X, y)
        acc_g = (t_gini.predict(X) == y).mean()
        acc_e = (t_ent.predict(X) == y).mean()
        assert acc_g > 0.85 and acc_e > 0.85

    def test_max_features_sqrt(self):
        X, y = make_multiclass()
        tree = DecisionTreeClassifier(max_features="sqrt", random_state=7)
        tree.fit(X, y)
        preds = tree.predict(X)
        assert len(preds) == len(y)

    def test_nan_in_features(self):
        X, y = make_linearly_separable(n=50)
        X_nan = X.copy()
        X_nan[::5, 0] = np.nan  # every 5th sample
        tree = DecisionTreeClassifier(max_depth=3, random_state=0)
        tree.fit(X_nan, y)
        # Should not crash
        preds = tree.predict(X_nan)
        assert len(preds) == 50

    def test_invalid_criterion_raises(self):
        with pytest.raises(ValueError):
            DecisionTreeClassifier(criterion="mse")

    def test_predict_before_fit_raises(self):
        tree = DecisionTreeClassifier()
        with pytest.raises(RuntimeError):
            tree.predict([[1.0, 2.0]])

    def test_single_sample_chunk(self):
        X, y = make_linearly_separable(n=40)
        tree = DecisionTreeClassifier(max_depth=3, random_state=0)
        for i in range(len(X)):
            tree.partial_fit(X[i:i+1], y[i:i+1])
        preds = tree.predict(X)
        assert len(preds) == len(X)

    def test_score_method(self):
        X, y = make_linearly_separable()
        tree = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X, y)
        score = tree.score(X, y)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# RandomForestClassifier
# ---------------------------------------------------------------------------

class TestRandomForestClassifier:

    def test_fit_predict(self):
        X, y = make_linearly_separable()
        rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0)
        rf.fit(X, y)
        acc = (rf.predict(X) == y).mean()
        assert acc > 0.90

    def test_partial_fit_streaming(self):
        X, y = make_multiclass()
        rf = RandomForestClassifier(n_estimators=5, max_depth=4, random_state=0)
        for chunk in np.array_split(np.column_stack([X, y]), 5):
            Xc, yc = chunk[:, :-1], chunk[:, -1].astype(int)
            rf.partial_fit(Xc, yc)
        acc = (rf.predict(X) == y).mean()
        assert acc > 0.7

    def test_predict_proba_shape(self):
        X, y = make_linearly_separable()
        rf = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y)
        proba = rf.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_classes_tracked(self):
        X, y = make_multiclass()
        rf = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y)
        assert len(rf.classes_) == 3

    def test_no_bootstrap(self):
        X, y = make_linearly_separable()
        rf = RandomForestClassifier(n_estimators=3, bootstrap=False, random_state=0)
        rf.fit(X, y)
        assert (rf.predict(X) == y).mean() > 0.8


# ---------------------------------------------------------------------------
# BaggingClassifier
# ---------------------------------------------------------------------------

class TestBaggingClassifier:

    def test_fit_predict(self):
        X, y = make_linearly_separable()
        base = DecisionTreeClassifier(max_depth=3, random_state=0)
        bag = BaggingClassifier(base_estimator=base, n_estimators=5, random_state=0)
        bag.partial_fit(X, y)
        acc = (bag.predict(X) == y).mean()
        assert acc > 0.80

    def test_default_base_estimator(self):
        X, y = make_linearly_separable()
        bag = BaggingClassifier(n_estimators=3, random_state=0)
        bag.partial_fit(X, y)
        preds = bag.predict(X)
        assert len(preds) == len(y)


# ---------------------------------------------------------------------------
# EnsembleClassifier
# ---------------------------------------------------------------------------

class TestEnsembleClassifier:

    def test_hard_voting(self):
        X, y = make_linearly_separable()
        trees = [
            DecisionTreeClassifier(max_depth=3, random_state=i)
            for i in range(3)
        ]
        ens = EnsembleClassifier(trees, voting="hard")
        ens.partial_fit(X, y)
        preds = ens.predict(X)
        assert len(preds) == len(y)

    def test_soft_voting(self):
        X, y = make_linearly_separable()
        trees = [
            DecisionTreeClassifier(max_depth=3, random_state=i)
            for i in range(3)
        ]
        ens = EnsembleClassifier(trees, voting="soft")
        ens.partial_fit(X, y)
        preds = ens.predict(X)
        assert len(preds) == len(y)

    def test_score_method(self):
        X, y = make_linearly_separable()
        trees = [DecisionTreeClassifier(max_depth=4, random_state=0)]
        ens = EnsembleClassifier(trees)
        ens.partial_fit(X, y)
        score = ens.score(X, y)
        assert 0.0 <= score <= 1.0