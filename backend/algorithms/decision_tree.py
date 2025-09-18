from __future__ import annotations

import math
import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
import graphviz

from utils.paths import DECISION_TREE_DIR, to_static_url


def calculate_entropy(data: List[Dict[str, Any]], target: str) -> float:
    if not data:
        return 0.0
    
    # Extract target values
    target_values = [row[target] for row in data]
    counts = Counter(target_values)
    total_samples = len(data)
    
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total_samples
            entropy -= p * np.log2(p)
    
    return entropy


def calculate_information_gain(data: List[Dict[str, Any]], feature: str, target: str, initial_entropy: float) -> float:
    if not data:
        return 0.0
    
    # Group data by feature values
    feature_groups: Dict[Any, List[Dict[str, Any]]] = {}
    for row in data:
        feature_value = row[feature]
        if feature_value not in feature_groups:
            feature_groups[feature_value] = []
        feature_groups[feature_value].append(row)
    
    # Calculate weighted entropy
    weighted_entropy = 0.0
    total_samples = len(data)
    
    for subset in feature_groups.values():
        p = len(subset) / total_samples
        subset_entropy = calculate_entropy(subset, target)
        weighted_entropy += p * subset_entropy
    
    return initial_entropy - weighted_entropy


def find_best_feature(data: List[Dict[str, Any]], features: List[str], target: str) -> Tuple[Optional[str], float]:
    if not data or not features:
        return None, 0.0
    
    initial_entropy = calculate_entropy(data, target)
    max_gain = -1.0
    best_feature = None
    
    for feature in features:
        gain = calculate_information_gain(data, feature, target, initial_entropy)
        if gain > max_gain:
            max_gain = gain
            best_feature = feature
    
    return best_feature, max_gain


def draw_decision_tree(tree: Any, filename: Optional[str] = None) -> Optional[str]:
    """Render a categorical (ID3) decision tree dictionary to an image and return the static URL."""

    dot = graphviz.Digraph(comment="Decision Tree", format="png")
    node_counter = 0

    def add_nodes_edges(sub_tree: Any) -> str:
        nonlocal node_counter
        current_id = str(node_counter)
        node_counter += 1

        if isinstance(sub_tree, dict) and sub_tree:
            feature_name = next(iter(sub_tree))
            dot.node(
                current_id,
                f"Feature: {feature_name}",
                shape="ellipse",
                style="filled",
                fillcolor="lightblue",
            )

            for value, child_tree in sub_tree[feature_name].items():
                child_id = add_nodes_edges(child_tree)
                dot.edge(current_id, child_id, label=str(value))
        else:
            dot.node(
                current_id,
                f"Result: {sub_tree}",
                shape="box",
                style="filled",
                fillcolor="lightgreen",
            )

        return current_id

    try:
        add_nodes_edges(tree)
        safe_name = filename or f"decision_tree_{uuid4().hex}"
        dot.render(filename=safe_name, directory=str(DECISION_TREE_DIR), cleanup=True)
        image_path = DECISION_TREE_DIR / f"{safe_name}.png"
        return to_static_url(image_path)
    except Exception:
        return None


# Legacy entropy function for compatibility
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


class ID3DecisionTree:

    def __init__(self, data: List[Dict[str, Any]], features: List[str], target: str):
        self.data = data
        self.features = features
        self.target = target
        self.tree = self.build_tree(self.data, self.features)
        self.initial_entropy = calculate_entropy(self.data, self.target)
    
    def build_tree(self, data: List[Dict[str, Any]], features: List[str]) -> Any:
        if not data:
            return Counter([row[self.target] for row in self.data]).most_common(1)[0][0]
        
        # Step 1: Check if all samples belong to one class
        target_values = [row[self.target] for row in data]
        unique_targets = list(set(target_values))
        
        if len(unique_targets) == 1:
            return unique_targets[0]
        
        # Step 2: If no more features to split on
        if not features:
            return Counter(target_values).most_common(1)[0][0]
        
        # Step 3: Find the best feature to split on
        best_feature, max_gain = find_best_feature(data, features, self.target)
        
        if best_feature is None or max_gain <= 0:
            return Counter(target_values).most_common(1)[0][0]
        
        # Step 4: Create tree structure
        tree = {best_feature: {}}
        remaining_features = [f for f in features if f != best_feature]
        
        # Get unique values for the best feature
        feature_values = list(set(row[best_feature] for row in data))
        
        # Step 5: Create subtree for each value of the best feature
        for value in feature_values:
            subset = [row for row in data if row[best_feature] == value]
            
            if subset:
                tree[best_feature][value] = self.build_tree(subset, remaining_features)
            else:
                # If subset is empty, use the most common class from parent
                tree[best_feature][value] = Counter(target_values).most_common(1)[0][0]
        
        return tree


def predict_with_tree(tree: Any, sample: Dict[str, Any]) -> Any:
    # If tree is not a dictionary, it's a leaf node
    if not isinstance(tree, dict):
        return tree
    
    # Get the feature name (root of current subtree)
    feature_name = list(tree.keys())[0]
    sub_tree = tree[feature_name]
    
    # Get the feature value from the sample
    feature_value = sample.get(feature_name)
    
    # Traverse to the appropriate subtree
    if feature_value in sub_tree:
        return predict_with_tree(sub_tree[feature_value], sample)
    else:
        # If feature value not seen during training, return default
        return "Unknown"


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
    
    @classmethod
    def from_id3_tree(cls, tree: Any, feature_name: Optional[str] = None) -> "Node":
        """Convert ID3 dictionary tree to Node structure."""
        if not isinstance(tree, dict):
            # Leaf node
            return cls(prediction=tree)
        
        # Get the feature name
        if feature_name is None:
            feature_name = list(tree.keys())[0]
        
        # Create children from dictionary
        children = {}
        for value, subtree in tree[feature_name].items():
            children[value] = cls.from_id3_tree(subtree)
        
        return cls(feature=feature_name, is_numeric=False, children=children)


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
    
    # Prepare data for ID3 algorithm
    # Combine X and y into a single dataset format
    if not X or not y:
        return Node(prediction="Unknown")
    
    # Add target variable to the dataset
    dataset = []
    target_name = "target"  # Internal target name
    
    for i, row in enumerate(X):
        combined_row = dict(row)
        combined_row[target_name] = y[i]
        dataset.append(combined_row)
    
    # Get feature names (excluding the target)
    features = [f for f in feature_types.keys()]
    
    # Check for numeric features - if any exist, use hybrid approach
    has_numeric = any(ftype == 'numeric' for ftype in feature_types.values())
    
    if has_numeric:
        # Use original algorithm for datasets with numeric features
        return _build_tree_original(X, y, feature_types, max_depth, min_samples_split)
    else:
        # Use pure ID3 algorithm for categorical features
        try:
            # Create ID3 decision tree
            id3_tree = ID3DecisionTree(dataset, features, target_name)
            
            # Convert ID3 tree to Node structure for API compatibility
            root_node = Node.from_id3_tree(id3_tree.tree)
            root_node.graph_image = draw_decision_tree(id3_tree.tree)
            root_node.algorithm = "ID3"
            root_node.initial_entropy = float(id3_tree.initial_entropy)
            root_node.raw_tree = id3_tree.tree

            return root_node
            
        except Exception as e:
            # Fallback to original algorithm if ID3 fails
            return _build_tree_original(X, y, feature_types, max_depth, min_samples_split)


def _build_tree_original(
    X: List[Dict[str, Any]],
    y: List[Any],
    feature_types: Dict[str, str],
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
        left_child = _build_tree_original(
            left_X, left_y, feature_types,
            None if max_depth is None else max_depth - 1,
            min_samples_split,
        )
        right_child = _build_tree_original(
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
            child = _build_tree_original(
                child_X, child_y, feature_types,
                None if max_depth is None else max_depth - 1,
                min_samples_split,
            )
            children[val] = child
        return Node(feature=best_feature, is_numeric=False, children=children)


def predict_tree(root: Node, row: Dict[str, Any]) -> Any:

    node = root
    
    # Handle prediction traversal
    while node.prediction is None:
        if node.is_numeric and node.threshold is not None:
            # Numeric split (original algorithm)
            try:
                v = float(row.get(node.feature, 0))
                if v <= node.threshold:
                    node = node.left  # type: ignore
                else:
                    node = node.right  # type: ignore
            except (ValueError, TypeError):
                # If conversion fails, return default
                return "Unknown"
        else:
            # Categorical split (ID3 algorithm)
            key = row.get(node.feature)
            if key in (node.children or {}):
                node = (node.children or {})[key]
            else:
                # Unseen category: fallback to most common prediction
                # Try to find a leaf node to get a reasonable default
                if node.children:
                    # Get first available child's prediction
                    first_child = next(iter(node.children.values()))
                    if first_child.prediction is not None:
                        return first_child.prediction
                return "Unknown"
    
    return node.prediction
