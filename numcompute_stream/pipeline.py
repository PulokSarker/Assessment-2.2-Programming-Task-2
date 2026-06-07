import numpy as np


class Pipeline:

    def __init__(self, steps):
        if not steps:
            raise ValueError("Pipeline requires at least one step.")
        self.steps = steps
        self._validate_steps()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_steps(self):
        for i, (name, est) in enumerate(self.steps[:-1]):
            if not (hasattr(est, "transform") and hasattr(est, "partial_fit")):
                raise ValueError(
                    f"Step '{name}' (index {i}) must have both .transform() and .partial_fit() methods."
                )
        last_name, last_est = self.steps[-1]
        if not hasattr(last_est, "predict"):
            raise ValueError(
                f"Final step '{last_name}' must have a .predict() method."
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def named_steps(self):
        return {name: est for name, est in self.steps}

    def __getitem__(self, name):
        return self.named_steps[name]

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def partial_fit(self, X, y=None, **fit_params):
        X = np.array(X, dtype=float)
        Xt = X

        for name, est in self.steps[:-1]:
            est.partial_fit(Xt)
            Xt = est.transform(Xt)

        last_name, last_est = self.steps[-1]
        if hasattr(last_est, "partial_fit"):
            last_est.partial_fit(Xt, y)
        else:
            last_est.fit(Xt, y)
        return self

    def fit(self, X, y=None):
        """Fit all steps (calls partial_fit once on full data)."""
        return self.partial_fit(X, y)

    def transform(self, X):
        """Apply all transformer steps (no final estimator)."""
        X = np.array(X, dtype=float)
        Xt = X
        for name, est in self.steps[:-1]:
            Xt = est.transform(Xt)
        return Xt

    def predict(self, X):
        """Transform X through all preprocessing steps, then predict."""
        Xt = self.transform(X)
        _, last_est = self.steps[-1]
        return last_est.predict(Xt)

    def predict_proba(self, X):
        """Transform X, then predict probabilities."""
        Xt = self.transform(X)
        _, last_est = self.steps[-1]
        if not hasattr(last_est, "predict_proba"):
            raise AttributeError("Final estimator has no predict_proba method.")
        return last_est.predict_proba(Xt)

    def score(self, X, y):
        """Accuracy score after transformation and prediction."""
        y_pred = self.predict(X)
        return float(np.mean(np.array(y_pred) == np.array(y)))

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self):
        steps_repr = ", ".join(f"('{n}', {type(e).__name__})" for n, e in self.steps)
        return f"Pipeline([{steps_repr}])"