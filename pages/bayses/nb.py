from collections import defaultdict, Counter
from typing import Dict, List, Tuple

class NaiveBayesCategorical:
    """
    Naive Bayes cho thuộc tính rời rạc (categorical) với làm trơn Laplace.
    Tham chiếu công thức P(C) và P(x_k|C) + Laplace smoothing. 
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.class_counts: Counter = Counter()
        self.feature_value_counts: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
        self.classes: List[str] = []
        self.features: List[str] = []
        self.feature_cardinality: Dict[str, int] = {}  # r_i cho mỗi thuộc tính
        self.N = 0  # số mẫu

    def fit(self, rows: List[Dict[str, str]], target_col: str):
        if not rows:
            raise ValueError("Tập dữ liệu rỗng.")
        # chuẩn hóa cột
        cols = list(rows[0].keys())
        if target_col not in cols:
            raise ValueError(f"Không tìm thấy cột nhãn '{target_col}' trong dữ liệu.")
        self.features = [c for c in cols if c != target_col]

        # đếm
        for r in rows:
            c = str(r[target_col]).strip()
            self.class_counts[c] += 1
            for f in self.features:
                v = str(r[f]).strip()
                self.feature_value_counts[f][c][v] += 1
            self.N += 1

        self.classes = list(self.class_counts.keys())

        # r_i (số giá trị rời rạc của thuộc tính f) tính trên toàn bộ dữ liệu
        for f in self.features:
            distinct_vals = set()
            for c in self.classes:
                distinct_vals.update(self.feature_value_counts[f][c].keys())
            self.feature_cardinality[f] = max(1, len(distinct_vals))

    def class_prior(self, c: str) -> float:
        # P(C) = (|C| + alpha) / (N + m*alpha)
        m = len(self.classes)
        return (self.class_counts[c] + self.alpha) / (self.N + m * self.alpha)

    def cond_prob(self, feature: str, value: str, c: str) -> float:
        # P(x_k = value | C=c) = (count(value in class c) + alpha) / (|C| + r_i*alpha)
        r_i = self.feature_cardinality[feature]
        count = self.feature_value_counts[feature][c][value]
        denom = self.class_counts[c] + r_i * self.alpha
        return (count + self.alpha) / denom

    def predict_proba(self, sample: Dict[str, str]) -> Dict[str, float]:
        # Tính hậu nghiệm tỷ lệ thuận: P(C) * Π_k P(x_k|C)
        numerators: Dict[str, float] = {}
        for c in self.classes:
            p = self.class_prior(c)
            for f in self.features:
                v = str(sample.get(f, "")).strip()
                # nếu giá trị chưa thấy, làm trơn Laplace sẽ tự xử lý
                p *= self.cond_prob(f, v, c)
            numerators[c] = p

        Z = sum(numerators.values()) or 1e-15
        return {c: numerators[c] / Z for c in self.classes}

    def explain(self, sample: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """
        Trả về chi tiết từng thành phần: prior và conditional của mỗi feature cho từng lớp
        """
        details: Dict[str, Dict[str, float]] = {}
        for c in self.classes:
            d = {}
            d[f"Prior P({c})"] = self.class_prior(c)
            for f in self.features:
                v = str(sample.get(f, "")).strip()
                d[f"P({f}={v}|{c})"] = self.cond_prob(f, v, c)
            details[c] = d
        return details