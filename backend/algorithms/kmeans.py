from __future__ import annotations

import random
from typing import List, Tuple, Optional
import numpy as np


def _generate_matrix(rows: int, cols: int, rng: random.Random) -> np.ndarray:

    if cols < rows:
        raise ValueError("Number of points must be >= number of clusters")

    # Initialize zero matrix for partitioning
    matrix = np.zeros((rows, cols), dtype=int)

    # Randomly assign one point to each cluster to ensure no empty clusters
    chosen_cols = rng.sample(range(cols), rows)
    for row in range(rows):
        matrix[row, chosen_cols[row]] = 1

    # Assign remaining points to random clusters
    for col in range(cols):
        if 1 not in matrix[:, col]:
            row = rng.randint(0, rows - 1)
            matrix[row, col] = 1
    
    return matrix


def kmeans(
    X: List[List[float]],
    k: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    init: str = "kmeans++",
    random_state: Optional[int] = None,
) -> Tuple[List[List[float]], List[int], float, int]:
    if k <= 0:
        raise ValueError("k must be > 0")
    if not X:
        return [], [], 0.0, 0

    # Convert to numpy array for efficient computation
    points = np.array(X)
    n = len(points)
    
    # Set up random number generator
    rng = random.Random(random_state)
    
    # Generate initial partition matrix
    ma_tran_phan_hoach = _generate_matrix(k, n, rng)
    
    # Main K-means iteration loop
    for it in range(1, max_iter + 1):
        # Calculate centroids for each cluster
        danh_sach_trong_tam = []
        
        for i in range(k):
            # Find indices of points belonging to cluster i
            idxs = np.where(ma_tran_phan_hoach[i] == 1)[0]
            
            if len(idxs) > 0:
                diem_trong_cum = points[idxs]  # Points in current cluster
                # Calculate centroid as mean of cluster points
                trong_tam = diem_trong_cum.mean(axis=0)
            else:
                # Empty cluster: reinitialize to a random point
                trong_tam = points[rng.randint(0, n - 1)].copy()
                
            danh_sach_trong_tam.append(trong_tam)
        
        # Convert centroids to numpy array
        centroids = np.array(danh_sach_trong_tam)
        
        # Calculate distances from each point to each centroid
        distances = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
        
        # Assign each point to the nearest centroid
        labels = np.argmin(distances, axis=1)

        # Create new partition matrix based on updated assignments
        ma_tran_phan_hoach_moi = np.zeros((k, n), dtype=int)
        for idx, label in enumerate(labels):
            ma_tran_phan_hoach_moi[label, idx] = 1
        
        # Check for convergence (no changes in assignments)
        if np.array_equal(ma_tran_phan_hoach, ma_tran_phan_hoach_moi):
            break
        else:
            ma_tran_phan_hoach = ma_tran_phan_hoach_moi

    # Calculate inertia (sum of squared distances to centroids)
    inertia = 0.0
    for i, point in enumerate(points):
        cluster_center = centroids[labels[i]]
        inertia += np.sum((point - cluster_center) ** 2)

    # Convert results to expected format
    centroids_list = [centroid.tolist() for centroid in centroids]
    labels_list = labels.tolist()

    return centroids_list, labels_list, float(inertia), it