# file: logistic_regression_gd.py
# author: Quintinasima Anyiam-Osigwe
# self-contained logistic regression (NumPy) with k-fold CV

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Iterable
import math
import numpy as np
import random

Array = np.ndarray

def _sigmoid(z: Array) -> Array:
    z = np.clip(z, -500, 500)            # avoid overflow in exp
    return 1.0 / (1.0 + np.exp(-z))

def _add_bias(X: Array) -> Array:
    return np.hstack([np.ones((X.shape[0], 1)), X])

@dataclass
class LRConfig:
    lr: float = 0.1
    epochs: int = 500
    l2: float = 1e-2
    tol: float = 1e-6
    verbose: bool = False
    random_state: int = 42

class LogisticRegressionGD:
    """L2-regularised logistic regression trained via batch gradient descent."""
    def __init__(self, config: LRConfig | None = None) -> None:
        self.config = config or LRConfig()
        self.coef_: Array | None = None

    def _loss_and_grad(self, Xb: Array, y: Array, w: Array) -> Tuple[float, Array]:
        p = _sigmoid(Xb @ w)
        eps = 1e-12
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        l2 = 0.5 * self.config.l2 * np.sum(w[1:] ** 2)   # don’t regularise bias
        loss += l2
        grad = (Xb.T @ (p - y)) / Xb.shape[0]
        grad[1:] += self.config.l2 * w[1:]
        return float(loss), grad

    def fit(self, X: Array, y: Array) -> "LogisticRegressionGD":
        assert set(np.unique(y)).issubset({0, 1}), "y must be binary {0,1}"
        rng = np.random.default_rng(self.config.random_state)
        Xb, w = _add_bias(X), rng.normal(scale=0.01, size=X.shape[1] + 1)
        prev = math.inf
        for epoch in range(self.config.epochs):
            loss, grad = self._loss_and_grad(Xb, y, w)
            w -= self.config.lr * grad
            if self.config.verbose and (epoch % 50 == 0 or epoch == self.config.epochs - 1):
                print(f"[epoch {epoch:04d}] loss={loss:.6f}")
            if abs(prev - loss) < self.config.tol:
                if self.config.verbose:
                    print(f"Early stopping at epoch {epoch}, Δ={abs(prev-loss):.3e}")
                break
            prev = loss
        self.coef_ = w
        return self

    def predict_proba(self, X: Array) -> Array:
        assert self.coef_ is not None, "Call fit() first."
        return _sigmoid(_add_bias(X) @ self.coef_)

    def predict(self, X: Array, threshold: float = 0.5) -> Array:
        return (self.predict_proba(X) >= threshold).astype(int)

def accuracy(y_true: Array, y_pred: Array) -> float:
    return float(np.mean(y_true == y_pred))

def precision_recall(y_true: Array, y_pred: Array) -> Tuple[float, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return float(prec), float(rec)

def kfold_indices(n: int, k: int, shuffle: bool = True, seed: int = 42):
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    folds = np.array_split(idx, k)
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, test_idx

def kfold_cv(X: Array, y: Array, k: int = 5, config: LRConfig | None = None):
    accs, precs, recs = [], [], []
    seed = (config.random_state if config else 42)
    for tr, te in kfold_indices(len(X), k=k, shuffle=True, seed=seed):
        m = LogisticRegressionGD(config=config)
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te], 0.5)
        accs.append(accuracy(y[te], pred))
        p, r = precision_recall(y[te], pred)
        precs.append(p); recs.append(r)
    return float(np.mean(accs)), float(np.mean(precs)), float(np.mean(recs))

# -------- tiny demo (lets reviewers run a quick smoke test) --------
def _make_synth(n: int = 600, seed: int = 0) -> Tuple[Array, Array]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    logits = 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.2
    y = (logits + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return X, y

def _quick_tests() -> None:
    random.seed(1)
    X, y = _make_synth(n=200, seed=1)
    cfg = LRConfig(lr=0.2, epochs=800, l2=1e-2, tol=1e-8, verbose=False, random_state=1)
    m = LogisticRegressionGD(cfg).fit(X, y)
    acc = accuracy(y, m.predict(X))
    assert 0.7 <= acc <= 1.0
    p = m.predict_proba(X[:5])
    assert p.shape == (5,) and np.all((p >= 0) & (p <= 1))
    a, pr, rc = kfold_cv(X, y, k=4, config=cfg)
    assert 0.6 <= a <= 1.0 and 0.5 <= pr <= 1.0 and 0.5 <= rc <= 1.0

if __name__ == "__main__":
    _quick_tests()
    X, y = _make_synth(n=600, seed=42)
    cfg = LRConfig(lr=0.15, epochs=700, l2=5e-3, tol=1e-7, verbose=False, random_state=42)
    acc, prec, rec = kfold_cv(X, y, k=5, config=cfg)
    print("Logistic Regression (from scratch) — 5-fold CV")
    print(f"accuracy: {acc:.3f} | precision: {prec:.3f} | recall: {rec:.3f}")
