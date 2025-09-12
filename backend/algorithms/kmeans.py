from __future__ import annotations

import math
import random
from typing import List, Tuple, Optional


def _squared_distance(a: List[float], b: List[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _mean(points: List[List[float]]) -> List[float]:
    if not points:
        return []
    n = len(points[0])
    acc = [0.0] * n
    for p in points:
        for i in range(n):
            acc[i] += p[i]
    return [v / len(points) for v in acc]


def _init_kmeans_plus_plus(X: List[List[float]], k: int, rng: random.Random) -> List[List[float]]:
    # Choose the first center uniformly at random
    centers = [list(rng.choice(X))]
    # Choose remaining centers with probability proportional to D(x)^2
    while len(centers) < k:
        distances = []
        for x in X:
            d2 = min(_squared_distance(x, c) for c in centers)
            distances.append(d2)
        total = sum(distances)
        if total == 0:
            # All points identical; just duplicate
            centers.append(list(centers[0]))
            continue
        r = rng.random() * total
        cum = 0.0
        idx = 0
        for i, d in enumerate(distances):
            cum += d
            if cum >= r:
                idx = i
                break
        centers.append(list(X[idx]))
    return centers


def kmeans(
    X: List[List[float]],
    k: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    init: str = "kmeans++",
    random_state: Optional[int] = None,
) -> Tuple[List[List[float]], List[int], float, int]:
    """
    Simple K-Means clustering from scratch.

    Returns (centroids, labels, inertia, n_iter)
    - centroids: list of k centroids
    - labels: cluster index for each sample
    - inertia: sum of squared distances to closest centroid
    - n_iter: iterations performed
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    if not X:
        return [], [], 0.0, 0
    n_features = len(X[0])
    rng = random.Random(random_state)

    # Initialize centers
    if init == "random":
        centers = [list(p) for p in rng.sample(X, k)]
    else:
        centers = _init_kmeans_plus_plus(X, k, rng)

    labels = [0] * len(X)
    for it in range(1, max_iter + 1):
        # Assign step
        for i, x in enumerate(X):
            best_idx = 0
            best_dist = _squared_distance(x, centers[0])
            for j in range(1, k):
                d = _squared_distance(x, centers[j])
                if d < best_dist:
                    best_dist = d
                    best_idx = j
            labels[i] = best_idx

        # Update step
        new_centers: List[List[float]] = []
        shift = 0.0
        for j in range(k):
            cluster_points = [X[i] for i, lab in enumerate(labels) if lab == j]
            if cluster_points:
                new_c = _mean(cluster_points)
            else:
                # Empty cluster: reinitialize to a random point
                new_c = list(rng.choice(X))
            shift = max(shift, math.sqrt(_squared_distance(new_c, centers[j])))
            new_centers.append(new_c)

        centers = new_centers
        if shift <= tol:
            break

    # Compute inertia
    inertia = 0.0
    for i, x in enumerate(X):
        inertia += _squared_distance(x, centers[labels[i]])

    return centers, labels, inertia, it

