from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def entropy(labels: List[Any]) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * math.log(p, 2)
    return ent


def information_gain(parent_labels: List[Any], splits: List[List[Any]]) -> float:
    H_parent = entropy(parent_labels)
    n = len(parent_labels)
    remainder = 0.0
    for split in splits:
        if not split:
            continue
        remainder += (len(split) / n) * entropy(split)
    return H_parent - remainder


@dataclass
class Node:
    # Leaf
    prediction: Optional[Any] = None
    # Split
    feature: Optional[str] = None
    threshold: Optional[float] = None
    is_numeric: bool = False
    children: Optional[Dict[Any, "Node"]] = None  # for categorical splits
    left: Optional["Node"] = None  # <= threshold
    right: Optional["Node"] = None  # > threshold

    def to_dict(self) -> Dict[str, Any]:
        if self.prediction is not None:
            return {"type": "leaf", "prediction": self.prediction}
        if self.is_numeric:
            return {
                "type": "split",
                "feature": self.feature,
                "is_numeric": True,
                "threshold": self.threshold,
                "left": self.left.to_dict() if self.left else None,
                "right": self.right.to_dict() if self.right else None,
            }
        else:
            return {
                "type": "split",
                "feature": self.feature,
                "is_numeric": False,
                "children": {k: v.to_dict() for k, v in (self.children or {}).items()},
            }


def majority_label(labels: List[Any]) -> Any:
    return Counter(labels).most_common(1)[0][0]


def _best_numeric_threshold(values: List[float], labels: List[Any]) -> Tuple[float, float]:
    # Return (best_gain, best_threshold)
    # Evaluate midpoints where label changes when sorting by value
    paired = sorted(zip(values, labels), key=lambda t: t[0])
    unique_pairs = []
    last_v = None
    for v, y in paired:
        if last_v is None or v != last_v:
            unique_pairs.append((v, y))
            last_v = v
        else:
            # keep last label for same value
            unique_pairs[-1] = (v, y)
    values_s = [v for v, _ in unique_pairs]
    labels_s = [y for _, y in unique_pairs]
    best_gain = -1.0
    best_thr = values_s[0]
    n = len(values_s)
    parent_labels = labels
    # Candidate thresholds between consecutive unique values
    for i in range(n - 1):
        if labels_s[i] == labels_s[i + 1]:
            continue
        thr = (values_s[i] + values_s[i + 1]) / 2
        left_labels = [y for v, y in paired if v <= thr]
        right_labels = [y for v, y in paired if v > thr]
        gain = information_gain(parent_labels, [left_labels, right_labels])
        if gain > best_gain:
            best_gain = gain
            best_thr = thr
    return best_gain, best_thr


def build_tree(
    X: List[Dict[str, Any]],
    y: List[Any],
    feature_types: Dict[str, str],  # 'numeric' or 'categorical'
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
) -> Node:
    # Stopping conditions
    if len(set(y)) == 1:
        return Node(prediction=y[0])
    if max_depth is not None and max_depth <= 0:
        return Node(prediction=majority_label(y))
    if len(y) < min_samples_split:
        return Node(prediction=majority_label(y))

    # Find best split
    best_gain = 0.0
    best_feature = None
    best_split = None
    best_is_numeric = False
    best_threshold = None

    for feature, ftype in feature_types.items():
        values = [row[feature] for row in X]
        if ftype == 'numeric':
            try:
                numeric_values = [float(v) for v in values]
            except Exception:
                continue
            gain, thr = _best_numeric_threshold(numeric_values, y)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_is_numeric = True
                best_threshold = thr
        else:
            # categorical split
            groups: Dict[Any, List[int]] = {}
            for idx, v in enumerate(values):
                groups.setdefault(v, []).append(idx)
            splits_labels = [[y[i] for i in idxs] for idxs in groups.values()]
            gain = information_gain(y, splits_labels)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_is_numeric = False
                best_split = groups

    if best_gain <= 0 or best_feature is None:
        return Node(prediction=majority_label(y))

    if best_is_numeric:
        thr = best_threshold  # type: ignore
        left_X, left_y, right_X, right_y = [], [], [], []
        for row, label in zip(X, y):
            v = float(row[best_feature])
            if v <= thr:
                left_X.append(row)
                left_y.append(label)
            else:
                right_X.append(row)
                right_y.append(label)
        left_child = build_tree(
            left_X, left_y, feature_types,
            None if max_depth is None else max_depth - 1,
            min_samples_split,
        )
        right_child = build_tree(
            right_X, right_y, feature_types,
            None if max_depth is None else max_depth - 1,
            min_samples_split,
        )
        return Node(
            feature=best_feature,
            threshold=thr,
            is_numeric=True,
            left=left_child,
            right=right_child,
        )
    else:
        children: Dict[Any, Node] = {}
        for val, idxs in (best_split or {}).items():
            child_X = [X[i] for i in idxs]
            child_y = [y[i] for i in idxs]
            child = build_tree(
                child_X, child_y, feature_types,
                None if max_depth is None else max_depth - 1,
                min_samples_split,
            )
            children[val] = child
        return Node(feature=best_feature, is_numeric=False, children=children)


def predict_tree(root: Node, row: Dict[str, Any]) -> Any:
    node = root
    while node.prediction is None:
        if node.is_numeric:
            v = float(row[node.feature])
            if v <= (node.threshold or 0.0):
                node = node.left  # type: ignore
            else:
                node = node.right  # type: ignore
        else:
            key = row.get(node.feature)
            if key in (node.children or {}):
                node = (node.children or {})[key]
            else:
                # unseen category: fallback
                return None
    return node.prediction

