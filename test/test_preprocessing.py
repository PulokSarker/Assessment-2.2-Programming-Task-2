import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from numcompute_stream.preprocessing import (
    StandardScaler, MinMaxScaler, Imputer, OneHotEncoder
)


class TestStandardScaler:

    def test_fit_transform_single_chunk(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        sc = StandardScaler()
        Xt = sc.fit_transform(X)
        np.testing.assert_allclose(Xt.mean(axis=0), [0, 0], atol=1e-10)
        np.testing.assert_allclose(Xt.std(axis=0, ddof=1), [1, 1], atol=1e-6)

    def test_incremental_matches_batch(self):
        rng = np.random.default_rng(10)
        X = rng.standard_normal((200, 4))
        # Batch
        sc_batch = StandardScaler()
        sc_batch.partial_fit(X)
        # Incremental
        sc_inc = StandardScaler()
        for chunk in np.array_split(X, 5):
            sc_inc.partial_fit(chunk)
        np.testing.assert_allclose(sc_inc.mean_, sc_batch.mean_, atol=1e-10)

    def test_zero_variance_feature(self):
        X = np.array([[1.0, 5.0], [1.0, 5.0], [1.0, 5.0]])
        sc = StandardScaler()
        Xt = sc.fit_transform(X)
        # Zero-variance features should not produce NaN or inf
        assert np.all(np.isfinite(Xt))

    def test_transform_before_fit_raises(self):
        sc = StandardScaler()
        with pytest.raises(RuntimeError):
            sc.transform([[1.0, 2.0]])

    def test_inverse_transform_roundtrip(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        sc = StandardScaler()
        Xt = sc.fit_transform(X)
        X_back = sc.inverse_transform(Xt)
        np.testing.assert_allclose(X_back, X, atol=1e-10)

    def test_1d_input(self):
        sc = StandardScaler()
        Xt = sc.fit_transform([1.0, 2.0, 3.0])
        assert Xt.shape == (3, 1)

    def test_nan_rows_skipped(self):
        X = np.array([[1.0], [np.nan], [3.0]])
        sc = StandardScaler()
        sc.partial_fit(X)
        # Mean of [1, 3] = 2
        np.testing.assert_allclose(sc.mean_, [2.0], atol=1e-10)


class TestMinMaxScaler:

    def test_scale_to_01(self):
        X = np.array([[0.0], [5.0], [10.0]])
        sc = MinMaxScaler()
        Xt = sc.fit_transform(X)
        np.testing.assert_allclose(Xt, [[0.0], [0.5], [1.0]])

    def test_incremental_min_max(self):
        sc = MinMaxScaler()
        sc.partial_fit([[0.0, 10.0]])
        sc.partial_fit([[5.0, 20.0]])
        sc.partial_fit([[-5.0, 5.0]])
        assert sc._data_min[0] == -5.0
        assert sc._data_max[1] == 20.0

    def test_custom_range(self):
        sc = MinMaxScaler(feature_range=(-1, 1))
        X = np.array([[0.0], [1.0]])
        Xt = sc.fit_transform(X)
        np.testing.assert_allclose(Xt, [[-1.0], [1.0]])

    def test_transform_before_fit_raises(self):
        sc = MinMaxScaler()
        with pytest.raises(RuntimeError):
            sc.transform([[1.0]])


class TestImputer:

    def test_mean_imputation(self):
        imp = Imputer(strategy="mean")
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        imp.partial_fit(X)
        Xt = imp.transform(X)
        # Mean of col0 = 2, col1 = 4
        assert Xt[0, 1] == pytest.approx(4.0)

    def test_median_imputation(self):
        imp = Imputer(strategy="median")
        X = np.array([[1.0], [3.0], [5.0], [np.nan]])
        imp.partial_fit(X)
        Xt = imp.transform(X)
        assert Xt[3, 0] == pytest.approx(3.0)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            Imputer(strategy="mode")

    def test_no_nans_unchanged(self):
        imp = Imputer()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        imp.partial_fit(X)
        Xt = imp.transform(X)
        np.testing.assert_allclose(Xt, X)

    def test_transform_before_fit_raises(self):
        imp = Imputer()
        with pytest.raises(RuntimeError):
            imp.transform([[np.nan]])


class TestOneHotEncoder:

    def test_basic_encoding(self):
        enc = OneHotEncoder()
        X = np.array([["a"], ["b"], ["c"]])
        enc.partial_fit(X)
        Xt = enc.transform([["a"], ["b"], ["c"]])
        assert Xt.shape == (3, 3)
        np.testing.assert_array_equal(Xt.sum(axis=1), [1, 1, 1])

    def test_incremental_vocab(self):
        enc = OneHotEncoder()
        enc.partial_fit([["a"], ["b"]])
        enc.partial_fit([["c"]])
        Xt = enc.transform([["a"], ["b"], ["c"]])
        assert Xt.shape[1] == 3

    def test_unknown_category_ignored(self):
        enc = OneHotEncoder(handle_unknown="ignore")
        enc.partial_fit([["a"], ["b"]])
        Xt = enc.transform([["z"]])
        np.testing.assert_array_equal(Xt, [[0.0, 0.0]])

    def test_unknown_category_raises(self):
        enc = OneHotEncoder(handle_unknown="error")
        enc.partial_fit([["a"]])
        with pytest.raises(ValueError):
            enc.transform([["z"]])

    def test_transform_before_fit_raises(self):
        enc = OneHotEncoder()
        with pytest.raises(RuntimeError):
            enc.transform([["a"]])