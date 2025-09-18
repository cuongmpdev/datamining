from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Mapping, Optional


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _round(value: float, digits: int = 6) -> float:
    return round(value, digits)


def _conditional_prob(
    count: int,
    class_total: int,
    unique_values: int,
    laplace: float,
) -> float:
    denom = class_total + laplace * unique_values
    return _safe_div(count + laplace, denom)


def compute_categorical_naive_bayes(
    rows: List[Dict[str, Any]],
    target: str,
    features: List[str],
    evidence: Optional[Mapping[str, Any]] = None,
    laplace: float = 0.0,
) -> Dict[str, Any]:
    if not features:
        raise ValueError("features list cannot be empty")

    evidence = dict(evidence or {})

    filtered_rows: List[Dict[str, Any]] = []
    class_counts: Counter = Counter()
    for row in rows:
        clazz = row.get(target)
        if clazz in (None, ""):
            continue
        filtered_rows.append(row)
        class_counts[clazz] += 1

    total_rows = sum(class_counts.values())
    classes = list(class_counts.keys())

    priors = {
        clazz: {
            "count": count,
            "prob": _round(_safe_div(count, total_rows)),
        }
        for clazz, count in class_counts.items()
    }

    conditional_counts: Dict[str, Dict[Any, Counter]] = {}
    for feature in features:
        value_map: Dict[Any, Counter] = defaultdict(Counter)
        for row in filtered_rows:
            clazz = row.get(target)
            value = row.get(feature)
            if clazz in (None, "") or value in (None, ""):
                continue
            value_map[value][clazz] += 1
        conditional_counts[feature] = value_map

    conditionals: Dict[str, Any] = {}
    for feature, value_map in conditional_counts.items():
        unique_count = len(value_map)
        per_value: Dict[Any, Dict[Any, Dict[str, float]]] = {}
        for value, counter in value_map.items():
            per_class: Dict[Any, Dict[str, float]] = {}
            for clazz in classes:
                count = counter.get(clazz, 0)
                prob_raw = _conditional_prob(count, class_counts[clazz], unique_count, laplace)
                per_class[clazz] = {
                    "count": count,
                    "prob": _round(prob_raw),
                }
            per_value[value] = per_class
        conditionals[feature] = {
            "unique_count": unique_count,
            "values": per_value,
        }

    evidence_filtered = {
        feature: value
        for feature, value in evidence.items()
        if feature in features and value not in (None, "")
    }

    posterior: Dict[Any, Dict[str, Any]] = {}
    score_sum = 0.0
    raw_scores: Dict[Any, float] = {}

    for clazz in classes:
        prior_raw = _safe_div(class_counts[clazz], total_rows)
        score = prior_raw
        components = []
        for feature in features:
            if feature not in evidence_filtered:
                continue
            value = evidence_filtered[feature]
            feature_info = conditionals.get(feature, {})
            value_info = feature_info.get("values", {}).get(value)
            unique_count = feature_info.get("unique_count", 0)
            count = 0
            if value_info is not None:
                count = value_info.get(clazz, {}).get("count", 0)
            prob_raw = _conditional_prob(count, class_counts[clazz], unique_count, laplace)
            score *= prob_raw
            components.append(
                {
                    "feature": feature,
                    "value": value,
                    "count": count,
                    "prob": _round(prob_raw),
                }
            )
        posterior[clazz] = {
            "prior": _round(prior_raw),
            "score": score,
            "components": components,
        }
        raw_scores[clazz] = score
        score_sum += score

    for clazz, info in posterior.items():
        posterior_prob = _safe_div(raw_scores.get(clazz, 0.0), score_sum)
        info["score"] = _round(raw_scores.get(clazz, 0.0))
        info["posterior"] = _round(posterior_prob)

    prediction = None
    if raw_scores:
        best_class, best_score = max(raw_scores.items(), key=lambda item: item[1])
        if best_score > 0.0:
            prediction = best_class

    return {
        "row_count": total_rows,
        "target": target,
        "features": features,
        "evidence": evidence_filtered,
        "priors": priors,
        "conditionals": conditionals,
        "posterior": posterior,
        "prediction": prediction,
        "classes": classes,
        "laplace": laplace,
    }

