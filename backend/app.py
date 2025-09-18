from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from utils.csv_utils import parse_csv_bytes, preview_summary, infer_types, select_columns
from algorithms.kmeans import kmeans
from algorithms.naive_bayes import compute_categorical_naive_bayes
from algorithms.decision_tree import build_tree, predict_tree
from algorithms.reduct import quick_reduct

app = FastAPI(title="Data Mining Algorithms API")

# Allow dev frontend at localhost:5173 by default
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/preview")
async def preview(file: UploadFile = File(...)) -> Dict[str, Any]:
    data = await file.read()
    rows, headers = parse_csv_bytes(data)
    return preview_summary(rows, headers)


@app.post("/api/kmeans")
async def api_kmeans(
    file: UploadFile = File(...),
    k: int = Form(...),
    max_iter: int = Form(100),
    tol: float = Form(1e-4),
    columns: Optional[str] = Form(None),  # comma-separated columns; if None use numeric inferred
    random_state: Optional[int] = Form(None),
) -> Dict[str, Any]:
    data = await file.read()
    rows, headers = parse_csv_bytes(data)
    types = infer_types(rows, headers)
    if columns:
        cols = [c.strip() for c in columns.split(",") if c.strip()]
    else:
        cols = [h for h in headers if types.get(h) == "numeric"]
    X = select_columns(rows, cols, numeric_only=True)
    centers, labels, inertia, n_iter = kmeans(X, k=k, max_iter=max_iter, tol=tol, random_state=random_state)
    return {
        "centroids": centers,
        "labels": labels,
        "inertia": inertia,
        "iterations": n_iter,
        "used_columns": cols,
        "row_count": len(X),
    }


@app.post("/api/naive-bayes")
async def api_naive_bayes(
    file: UploadFile = File(...),
    target: str = Form(...),
    features: Optional[str] = Form(None),
    evidence: Optional[str] = Form(None),
    laplace: float = Form(0.0),
) -> Dict[str, Any]:
    data = await file.read()
    rows, headers = parse_csv_bytes(data)

    if target not in headers:
        raise HTTPException(status_code=400, detail="Target column not found in dataset")

    feature_list: List[str]
    if features:
        parsed_features: List[str] = []
        try:
            loaded = json.loads(features)
            if isinstance(loaded, list):
                parsed_features = [str(item) for item in loaded]
        except json.JSONDecodeError:
            parsed_features = [c.strip() for c in features.split(",") if c.strip()]
        feature_list = [f for f in parsed_features if f in headers and f != target]
    else:
        feature_list = [h for h in headers if h != target]

    if not feature_list:
        raise HTTPException(status_code=400, detail="No feature columns selected")

    evidence_map: Dict[str, Any] = {}
    if evidence:
        try:
            loaded_evidence = json.loads(evidence)
            if isinstance(loaded_evidence, dict):
                for key, value in loaded_evidence.items():
                    if key in headers:
                        evidence_map[key] = "" if value is None else str(value)
        except json.JSONDecodeError:
            pass

    result = compute_categorical_naive_bayes(
        rows,
        target=target,
        features=feature_list,
        evidence=evidence_map,
        laplace=laplace,
    )

    return result


@app.post("/api/decision-tree")
async def api_decision_tree(
    file: UploadFile = File(...),
    target: str = Form(...),
    max_depth: Optional[int] = Form(None),
    min_samples_split: int = Form(2),
) -> Dict[str, Any]:
    data = await file.read()
    rows, headers = parse_csv_bytes(data)
    # Build feature types (numeric vs categorical) excluding target
    types_all = infer_types(rows, headers)
    feature_types = {h: t for h, t in types_all.items() if h != target}
    # Prepare X (as dict) and y
    X = [{h: r.get(h) for h in feature_types.keys()} for r in rows]
    y = [r.get(target) for r in rows]
    root = build_tree(X, y, feature_types, max_depth=max_depth, min_samples_split=min_samples_split)
    # Evaluate on training set
    preds = [predict_tree(root, row) for row in X]
    acc = sum(1 for p, t in zip(preds, y) if p == t) / len(y) if y else 0.0
    return {
        "tree": root.to_dict(),
        "accuracy_train": acc,
        "target": target,
        "feature_types": feature_types,
    }


@app.post("/api/reduct")
async def api_reduct(
    file: UploadFile = File(...),
    decision: str = Form(...),
    conditional: Optional[str] = Form(None),  # comma-separated; if None, use all except decision
) -> Dict[str, Any]:
    data = await file.read()
    rows, headers = parse_csv_bytes(data)
    if conditional:
        cond_attrs = [c.strip() for c in conditional.split(",") if c.strip()]
    else:
        cond_attrs = [h for h in headers if h != decision]
    result = quick_reduct(rows, cond_attrs, decision)
    return {
        **result,
        "decision": decision,
        "conditional": cond_attrs,
        "row_count": len(rows),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
