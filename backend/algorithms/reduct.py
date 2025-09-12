from __future__ import annotations

from typing import Any, Dict, List, Tuple, Set


def equivalence_classes(rows: List[Dict[str, Any]], attrs: List[str]) -> Dict[Tuple[Any, ...], List[int]]:
    groups: Dict[Tuple[Any, ...], List[int]] = {}
    for idx, row in enumerate(rows):
        key = tuple(row[a] for a in attrs)
        groups.setdefault(key, []).append(idx)
    return groups


def positive_region(rows: List[Dict[str, Any]], cond_attrs: List[str], decision_attr: str) -> Set[int]:
    pos: Set[int] = set()
    if not cond_attrs:
        return pos
    cond_classes = equivalence_classes(rows, cond_attrs)
    for _, idxs in cond_classes.items():
        decisions = set(rows[i][decision_attr] for i in idxs)
        if len(decisions) == 1:
            pos.update(idxs)
    return pos


def dependency_degree(rows: List[Dict[str, Any]], cond_attrs: List[str], decision_attr: str) -> float:
    if not rows:
        return 0.0
    pos = positive_region(rows, cond_attrs, decision_attr)
    return len(pos) / len(rows)


def quick_reduct(rows: List[Dict[str, Any]], cond_attrs: List[str], decision_attr: str) -> Dict[str, Any]:
    """
    QuickReduct algorithm (greedy) for Rough Set reduct.
    Returns dict with keys: reduct, order, gamma_R, gamma_C
    """
    C = list(cond_attrs)
    R: List[str] = []
    gamma_C = dependency_degree(rows, C, decision_attr)
    gamma_R = 0.0
    order: List[Tuple[str, float]] = []

    # Greedily add attributes that maximize dependency increase
    while gamma_R < gamma_C:
        best_attr = None
        best_gamma = gamma_R
        for a in C:
            if a in R:
                continue
            g = dependency_degree(rows, R + [a], decision_attr)
            if g > best_gamma:
                best_gamma = g
                best_attr = a
        if best_attr is None:
            break
        R.append(best_attr)
        gamma_R = best_gamma
        order.append((best_attr, gamma_R))

    # Optional minimality check: try removing redundant attributes
    changed = True
    while changed and len(R) > 1:
        changed = False
        for a in list(R):
            R_try = [x for x in R if x != a]
            if dependency_degree(rows, R_try, decision_attr) >= gamma_R - 1e-12:
                R = R_try
                changed = True
                break

    return {
        "reduct": R,
        "order": order,
        "gamma_R": gamma_R,
        "gamma_C": gamma_C,
    }

