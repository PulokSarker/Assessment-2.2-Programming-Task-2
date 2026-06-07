import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import RandomForestClassifier
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class TestPipeline:

    def _make_pipeline(self):
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=3, random_state=0)),
        ])

    def test_partial_fit_predict(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 4))
        y = (X[:, 0] > 0).astype(int)
        pipe = self._make_pipeline()
        pipe.partial_fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == 100

    def test_streaming_chunks(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((200, 3))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        pipe = self._make_pipeline()
        for chunk in np.array_split(np.column_stack([X, y]), 4):
            Xc, yc = chunk[:, :-1], chunk[:, -1].astype(int)
            pipe.partial_fit(Xc, yc)
        acc = pipe.score(X, y)
        assert 0.0 <= acc <= 1.0

    def test_empty_steps_raises(self):
        with pytest.raises(ValueError):
            Pipeline([])

    def test_invalid_transformer_raises(self):
        class BadTransformer:
            def partial_fit(self, X): return self
            # Missing .transform

        with pytest.raises(ValueError):
            Pipeline([("bad", BadTransformer()), ("model", DecisionTreeClassifier())])

    def test_predict_proba(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 2))
        y = (X[:, 0] > 0).astype(int)
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=3, random_state=0)),
        ])
        pipe.partial_fit(X, y)
        proba = pipe.predict_proba(X)
        assert proba.shape[0] == 50

    def test_named_steps_access(self):
        pipe = self._make_pipeline()
        assert "scale" in pipe.named_steps
        assert "model" in pipe.named_steps

    def test_repr(self):
        pipe = self._make_pipeline()
        r = repr(pipe)
        assert "Pipeline" in r

    def test_rf_in_pipeline(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((100, 3))
        y = (X.sum(axis=1) > 0).astype(int)
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=3, random_state=0)),
        ])
        pipe.partial_fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == 100


# ---------------------------------------------------------------------------
# StreamTrainer
# ---------------------------------------------------------------------------

class TestStreamTrainer:

    def _setup(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((500, 4))
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
        model = DecisionTreeClassifier(max_depth=4, random_state=0)
        trainer = StreamTrainer(model)
        return trainer, X, y

    def test_fit_chunk_logs_entry(self):
        trainer, X, y = self._setup()
        trainer.fit_chunk(X[:50], y[:50])
        assert len(trainer.chunk_logs_) == 1
        log = trainer.chunk_logs_[0]
        assert "accuracy" in log
        assert "cumulative_accuracy" in log
        assert "fit_time_s" in log

    def test_multiple_chunks_logged(self):
        trainer, X, y = self._setup()
        for i in range(5):
            trainer.fit_chunk(X[i*100:(i+1)*100], y[i*100:(i+1)*100])
        assert len(trainer.chunk_logs_) == 5

    def test_score_chunk(self):
        trainer, X, y = self._setup()
        trainer.fit_chunk(X[:200], y[:200])
        score = trainer.score_chunk(X[200:300], y[200:300])
        assert 0.0 <= score <= 1.0

    def test_get_metric_history(self):
        trainer, X, y = self._setup()
        for i in range(3):
            trainer.fit_chunk(X[i*100:(i+1)*100], y[i*100:(i+1)*100])
        history = trainer.get_metric_history("accuracy")
        assert len(history) == 3
        assert all(0.0 <= v <= 1.0 for v in history)

    def test_make_chunks_splits_correctly(self):
        X = np.arange(100).reshape(-1, 2).astype(float)
        y = np.zeros(50)
        chunks = list(StreamTrainer.make_chunks(X, y, n_chunks=5))
        assert len(chunks) == 5
        total = sum(len(xc) for xc, yc in chunks)
        assert total == 50

    def test_run_stream(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((300, 3))
        y = (X[:, 0] > 0).astype(int)
        model = DecisionTreeClassifier(max_depth=3, random_state=0)
        trainer = StreamTrainer(model)
        trainer.run_stream(X, y, n_chunks=6)
        assert len(trainer.chunk_logs_) == 6

    def test_predict_after_fit(self):
        trainer, X, y = self._setup()
        trainer.fit_chunk(X[:100], y[:100])
        preds = trainer.predict(X[100:150])
        assert len(preds) == 50

    def test_cumulative_accuracy_increases_on_easy_data(self):
        """On clean, separable data accuracy should generally be high."""
        rng = np.random.default_rng(99)
        X0 = rng.standard_normal((200, 2)) + np.array([-5, -5])
        X1 = rng.standard_normal((200, 2)) + np.array([5, 5])
        X = np.vstack([X0, X1])
        y = np.array([0] * 200 + [1] * 200)
        model = DecisionTreeClassifier(max_depth=3, random_state=0)
        trainer = StreamTrainer(model)
        trainer.run_stream(X, y, n_chunks=4)
        final_cum_acc = trainer.chunk_logs_[-1]["cumulative_accuracy"]
        assert final_cum_acc > 0.80

    def test_with_scaler(self):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((200, 3)) * 100  # large scale
        y = (X[:, 0] > 0).astype(int)
        model = DecisionTreeClassifier(max_depth=3, random_state=0)
        scaler = StandardScaler()
        trainer = StreamTrainer(model, scaler=scaler)
        trainer.fit_chunk(X, y)
        assert len(trainer.chunk_logs_) == 1


