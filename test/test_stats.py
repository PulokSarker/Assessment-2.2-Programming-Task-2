import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from numcompute_stream.stats import (
    StreamingStats, chunk_mean, chunk_variance, chunk_quantiles, chunk_histogram
)


class TestStreamingStats:

    def test_single_chunk_mean(self):
        ss = StreamingStats(n_features=2)
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        ss.update_stats(X)
        np.testing.assert_allclose(ss.mean(), [3.0, 4.0])

    def test_incremental_mean_matches_batch(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((300, 3))
        batch_mean = X.mean(axis=0)

        ss = StreamingStats(n_features=3)
        for chunk in np.array_split(X, 10):
            ss.update_stats(chunk)
        np.testing.assert_allclose(ss.mean(), batch_mean, atol=1e-10)

    def test_variance_two_pass(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((500, 2))
        ss = StreamingStats(n_features=2)
        ss.update_stats(X)
        np.testing.assert_allclose(ss.variance(ddof=1), X.var(axis=0, ddof=1), atol=1e-6)

    def test_nan_ignored(self):
        X = np.array([[1.0, np.nan], [np.nan, 2.0], [3.0, 4.0]])
        ss = StreamingStats(n_features=2)
        ss.update_stats(X)
        # Only row [3, 4] has no NaN – only it counts for Welford
        assert ss.count() == 1

    def test_count_accumulates(self):
        ss = StreamingStats(n_features=1)
        ss.update_stats([[1.0], [2.0]])
        ss.update_stats([[3.0]])
        assert ss.count() == 3

    def test_min_max(self):
        ss = StreamingStats(n_features=2)
        ss.update_stats([[1.0, 5.0], [3.0, 2.0]])
        np.testing.assert_array_equal(ss.min(), [1.0, 2.0])
        np.testing.assert_array_equal(ss.max(), [3.0, 5.0])

    def test_quantiles(self):
        X = np.arange(100).reshape(-1, 1).astype(float)
        ss = StreamingStats(n_features=1)
        ss.update_stats(X)
        q = ss.quantiles(q=(0.5,))
        np.testing.assert_allclose(q[0.5], [49.5], atol=1.0)

    def test_histogram_returns_correct_shape(self):
        X = np.random.default_rng(2).standard_normal((200, 2))
        ss = StreamingStats(n_features=2)
        ss.update_stats(X)
        counts, edges = ss.histogram(feature_idx=0, bins=10)
        assert len(counts) == 10
        assert len(edges) == 11

    def test_sliding_window(self):
        ss = StreamingStats(n_features=1, sliding_window=5)
        for i in range(20):
            ss.update_stats([[float(i)]])
        # Buffer should only hold last 5
        assert len(ss._buffer) == 5

    def test_reset_clears_state(self):
        ss = StreamingStats(n_features=1)
        ss.update_stats([[1.0], [2.0]])
        ss.reset()
        assert ss.count() == 0
        np.testing.assert_array_equal(ss.mean(), [0.0])

    def test_zero_variance_chunk(self):
        X = np.ones((10, 2))
        ss = StreamingStats(n_features=2)
        ss.update_stats(X)
        np.testing.assert_allclose(ss.variance(), [0.0, 0.0], atol=1e-10)

    def test_summary_keys(self):
        ss = StreamingStats(n_features=2)
        ss.update_stats([[1.0, 2.0]])
        s = ss.summary()
        assert set(s.keys()) >= {"n", "mean", "std", "min", "max"}


class TestChunkFunctions:

    def test_chunk_mean(self):
        X = [[1, 2], [3, 4]]
        np.testing.assert_allclose(chunk_mean(X), [2.0, 3.0])

    def test_chunk_variance(self):
        X = [[1, 2], [3, 4], [5, 6]]
        expected = np.var([[1, 2], [3, 4], [5, 6]], axis=0, ddof=1)
        np.testing.assert_allclose(chunk_variance(X, ddof=1), expected, atol=1e-10)

    def test_chunk_quantiles_keys(self):
        X = np.arange(20).reshape(10, 2).astype(float)
        q = chunk_quantiles(X, q=(0.25, 0.75))
        assert 0.25 in q and 0.75 in q

    def test_chunk_histogram_empty(self):
        counts, edges = chunk_histogram([[np.nan]], feature_idx=0, bins=5)
        assert len(counts) == 0