from __future__ import annotations

import math
from collections import defaultdict
from typing import List, Dict, Any, Tuple


class GaussianNB:
    def __init__(self):
        self.classes_: List[Any] = []
        self.class_prior_: Dict[Any, float] = {}
        self.theta_: Dict[Any, List[float]] = {}
        self.sigma_: Dict[Any, List[float]] = {}

    def fit(self, X: List[List[float]], y: List[Any]):
        n = len(X)
        if n == 0:
            return self
        n_features = len(X[0])
        by_class: Dict[Any, List[List[float]]] = defaultdict(list)
        for xi, yi in zip(X, y):
            by_class[yi].append(xi)

        self.classes_ = list(by_class.keys())
        for c, rows in by_class.items():
            self.class_prior_[c] = len(rows) / n
            # mean
            mu = [0.0] * n_features
            for row in rows:
                for j in range(n_features):
                    mu[j] += row[j]
            mu = [v / len(rows) for v in mu]
            self.theta_[c] = mu
            # variance (with small epsilon to avoid zero)
            var = [0.0] * n_features
            for row in rows:
                for j in range(n_features):
                    diff = row[j] - mu[j]
                    var[j] += diff * diff
            var = [v / max(1, (len(rows) - 1)) for v in var]
            # add epsilon for numerical stability
            self.sigma_[c] = [v if v > 1e-9 else 1e-9 for v in var]
        return self

    @staticmethod
    def _log_gaussian_pdf(x: float, mu: float, var: float) -> float:
        # log of normal pdf
        return -0.5 * (math.log(2 * math.pi * var) + ((x - mu) ** 2) / var)

    def predict_log_proba(self, X: List[List[float]]) -> List[Dict[Any, float]]:
        results: List[Dict[Any, float]] = []
        for xi in X:
            lp: Dict[Any, float] = {}
            for c in self.classes_:
                s = math.log(self.class_prior_.get(c, 1e-12))
                mu = self.theta_[c]
                var = self.sigma_[c]
                for j in range(len(xi)):
                    s += self._log_gaussian_pdf(xi[j], mu[j], var[j])
                lp[c] = s
            results.append(lp)
        return results

    def predict(self, X: List[List[float]]) -> List[Any]:
        preds: List[Any] = []
        for lp in self.predict_log_proba(X):
            # choose argmax class
            best_c = None
            best_s = -float("inf")
            for c, s in lp.items():
                if s > best_s:
                    best_s = s
                    best_c = c
            preds.append(best_c)
        return preds


class MultinomialNB:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes_: List[Any] = []
        self.class_prior_: Dict[Any, float] = {}
        self.feature_log_prob_: Dict[Any, List[float]] = {}
        self.feature_count_: Dict[Any, List[float]] = {}
        self.feature_sum_: Dict[Any, float] = {}

    def fit(self, X: List[List[float]], y: List[Any]):
        n = len(X)
        if n == 0:
            return self
        n_features = len(X[0])
        by_class: Dict[Any, List[List[float]]] = defaultdict(list)
        for xi, yi in zip(X, y):
            by_class[yi].append(xi)
        self.classes_ = list(by_class.keys())

        for c, rows in by_class.items():
            self.class_prior_[c] = len(rows) / n
            fc = [0.0] * n_features
            for row in rows:
                for j in range(n_features):
                    v = row[j]
                    if v < 0:
                        raise ValueError("MultinomialNB requires non-negative feature values.")
                    fc[j] += v
            self.feature_count_[c] = fc
            total = sum(fc)
            self.feature_sum_[c] = total
            # Laplace smoothing
            denom = total + self.alpha * n_features
            probs = [ (fc[j] + self.alpha) / denom for j in range(n_features) ]
            self.feature_log_prob_[c] = [ math.log(p) for p in probs ]
        return self

    def predict_log_proba(self, X: List[List[float]]) -> List[Dict[Any, float]]:
        results: List[Dict[Any, float]] = []
        for xi in X:
            lp: Dict[Any, float] = {}
            for c in self.classes_:
                s = math.log(self.class_prior_.get(c, 1e-12))
                flp = self.feature_log_prob_[c]
                for j, xij in enumerate(xi):
                    s += xij * flp[j]
                lp[c] = s
            results.append(lp)
        return results

    def predict(self, X: List[List[float]]) -> List[Any]:
        preds: List[Any] = []
        for lp in self.predict_log_proba(X):
            best_c = None
            best_s = -float("inf")
            for c, s in lp.items():
                if s > best_s:
                    best_s = s
                    best_c = c
            preds.append(best_c)
        return preds

