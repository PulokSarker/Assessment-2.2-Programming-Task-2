"""
tests/test_metrics.py – Unit tests for numcompute_stream.metrics

Covers: accuracy (streaming, rolling), confusion matrix, precision/recall/f1,
        AUC, reset, shape mismatch errors, binary/multiclass.
"""

import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from numcompute_stream.metrics import (
    StreamingAccuracy, StreamingConfusionMatrix,
    StreamingPrecisionRecallF1, StreamingAUC,
    accuracy_score, precision_recall_f1, confusion_matrix,
)


class TestStreamingAccuracy:

    def test_perfect_accuracy(self):
        acc = StreamingAccuracy()
        acc.update([0, 1, 2], [0, 1, 2])
        assert acc.result() == 1.0

    def test_zero_accuracy(self):
        acc = StreamingAccuracy()
        acc.update([0, 0, 0], [1, 1, 1])
        assert acc.result() == 0.0

    def test_partial_accuracy(self):
        acc = StreamingAccuracy()
        acc.update([0, 1, 0, 1], [0, 0, 0, 1])
        assert acc.result() == pytest.approx(0.75)

    def test_cumulative_across_chunks(self):
        acc = StreamingAccuracy()
        acc.update([0, 1], [0, 1])   # 2/2
        acc.update([0, 1], [0, 0])   # 1/2
        assert acc.result() == pytest.approx(0.75)

    def test_rolling_window(self):
        acc = StreamingAccuracy(rolling_window=4)
        acc.update([0, 0, 0, 0], [0, 0, 0, 0])   # 4/4 correct
        acc.update([0, 0, 0, 0], [1, 1, 1, 1])   # 0/4 correct
        # Rolling window = 4 → only last 4 samples matter
        assert acc.result() == pytest.approx(0.0)

    def test_shape_mismatch_raises(self):
        acc = StreamingAccuracy()
        with pytest.raises(ValueError):
            acc.update([0, 1], [0, 1, 2])

    def test_reset(self):
        acc = StreamingAccuracy()
        acc.update([0, 1], [0, 1])
        acc.reset()
        assert acc.result() == 0.0

    def test_empty_result(self):
        acc = StreamingAccuracy()
        assert acc.result() == 0.0


class TestStreamingConfusionMatrix:

    def test_binary_matrix(self):
        cm = StreamingConfusionMatrix(classes=[0, 1])
        cm.update([0, 1, 0, 1], [0, 1, 1, 0])
        mat, classes = cm.result()
        assert mat.shape == (2, 2)
        assert mat[0, 0] == 1   # TN
        assert mat[1, 1] == 1   # TP
        assert mat[0, 1] == 1   # FP
        assert mat[1, 0] == 1   # FN

    def test_incremental_accumulation(self):
        cm = StreamingConfusionMatrix(classes=[0, 1, 2])
        cm.update([0, 1], [0, 2])
        cm.update([2, 1], [2, 1])
        mat, _ = cm.result()
        assert mat.sum() == 4

    def test_reset(self):
        cm = StreamingConfusionMatrix()
        cm.update([0, 1], [0, 1])
        cm.reset()
        mat, classes = cm.result()
        assert mat.sum() == 0


class TestStreamingPrecisionRecallF1:

    def test_perfect_scores(self):
        m = StreamingPrecisionRecallF1(classes=[0, 1])
        m.update([0, 1, 0, 1], [0, 1, 0, 1])
        res = m.result()
        assert res["macro_precision"] == pytest.approx(1.0)
        assert res["macro_recall"] == pytest.approx(1.0)
        assert res["macro_f1"] == pytest.approx(1.0)

    def test_all_wrong(self):
        m = StreamingPrecisionRecallF1(classes=[0, 1])
        m.update([0, 0], [1, 1])
        res = m.result()
        assert res["macro_recall"] == pytest.approx(0.0)

    def test_result_keys(self):
        m = StreamingPrecisionRecallF1()
        m.update([0, 1], [0, 1])
        res = m.result()
        assert "macro_f1" in res and "macro_precision" in res and "macro_recall" in res

    def test_multiclass(self):
        m = StreamingPrecisionRecallF1(classes=[0, 1, 2])
        m.update([0, 1, 2, 0, 1, 2], [0, 1, 2, 1, 2, 0])
        res = m.result()
        assert 0.0 <= res["macro_f1"] <= 1.0


class TestStreamingAUC:

    def test_perfect_binary_auc(self):
        auc = StreamingAUC()
        auc.update([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        assert auc.result() == pytest.approx(1.0, abs=0.01)

    def test_random_auc_near_half(self):
        rng = np.random.default_rng(5)
        y_true = rng.integers(0, 2, 100)
        y_score = rng.random(100)
        auc = StreamingAUC()
        auc.update(y_true, y_score)
        assert 0.3 <= auc.result() <= 0.7

    def test_undefined_when_single_class(self):
        auc = StreamingAUC()
        auc.update([0, 0, 0], [0.1, 0.5, 0.9])
        assert auc.result() == pytest.approx(0.5)

    def test_rolling_window_auc(self):
        auc = StreamingAUC(rolling_window=4)
        auc.update([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        r1 = auc.result()
        auc.update([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
        r2 = auc.result()
        # Both should be valid probabilities
        assert 0.0 <= r2 <= 1.0


class TestConvenienceFunctions:

    def test_accuracy_score(self):
        assert accuracy_score([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)
        assert accuracy_score([0, 1, 2], [2, 1, 0]) == pytest.approx(1/3)

    def test_precision_recall_f1(self):
        p, r, f = precision_recall_f1([0, 1, 0, 1], [0, 1, 0, 1], average="macro")
        assert p == pytest.approx(1.0)

    def test_confusion_matrix_shape(self):
        mat, classes = confusion_matrix([0, 1, 2], [0, 1, 1])
        assert mat.shape == (3, 3)