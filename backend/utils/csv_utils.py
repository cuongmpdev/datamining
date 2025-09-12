from __future__ import annotations

import csv
import io
from typing import Any, Dict, List, Tuple


def sniff_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample)
        return dialect.delimiter
    except Exception:
        return ","


def parse_csv_bytes(data: bytes) -> Tuple[List[Dict[str, Any]], List[str]]:
    text = data.decode("utf-8", errors="ignore")
    sample = text[: 1024]
    delimiter = sniff_delimiter(sample)
    f = io.StringIO(text)
    reader = csv.DictReader(f, delimiter=delimiter)
    rows = [dict(row) for row in reader]
    headers = reader.fieldnames or []
    return rows, headers


def infer_types(rows: List[Dict[str, Any]], headers: List[str]) -> Dict[str, str]:
    types: Dict[str, str] = {}
    for h in headers:
        is_numeric = True
        for row in rows:
            v = row.get(h, "")
            if v is None or v == "":
                continue
            try:
                float(v)
            except Exception:
                is_numeric = False
                break
        types[h] = "numeric" if is_numeric else "categorical"
    return types


def select_columns(
    rows: List[Dict[str, Any]],
    columns: List[str],
    numeric_only: bool = False,
) -> List[List[float]]:
    X: List[List[float]] = []
    for row in rows:
        vec: List[float] = []
        for c in columns:
            v = row.get(c, "")
            if v == "" or v is None:
                vec.append(0.0)
            else:
                try:
                    vec.append(float(v))
                except Exception:
                    if numeric_only:
                        # skip bad row entirely
                        vec = []
                        break
                    else:
                        # map categorical to hash bucket as a fallback numeric
                        vec.append(float(abs(hash(v)) % 1000))
        if vec:
            X.append(vec)
    return X


def split_X_y(
    rows: List[Dict[str, Any]],
    feature_cols: List[str],
    target_col: str,
    cast_numeric: bool = True,
) -> Tuple[List[List[float]], List[Any]]:
    X: List[List[float]] = []
    y: List[Any] = []
    for row in rows:
        xi: List[float] = []
        skip = False
        for c in feature_cols:
            v = row.get(c, "")
            if cast_numeric:
                if v == "" or v is None:
                    xi.append(0.0)
                else:
                    try:
                        xi.append(float(v))
                    except Exception:
                        # treat categorical by hashing into numeric bin for GaussianNB/DT numeric splits
                        xi.append(float(abs(hash(v)) % 1000))
            else:
                # keep as string
                xi.append(v)
        if skip:
            continue
        X.append(xi)
        y.append(row.get(target_col))
    return X, y


def preview_summary(rows: List[Dict[str, Any]], headers: List[str], limit: int = 10) -> Dict[str, Any]:
    sample = rows[:limit]
    types = infer_types(rows, headers)
    uniques = {}
    for h in headers:
        s = set()
        for row in rows[:1000]:  # cap for speed
            v = row.get(h)
            s.add(v)
        uniques[h] = len(s)
    return {
        "headers": headers,
        "types": types,
        "uniques": uniques,
        "sample": sample,
        "row_count": len(rows),
    }

