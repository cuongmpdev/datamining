import os
import graphviz
from flask import Flask, render_template, request, Blueprint, url_for
import pandas as pd
import numpy as np
from collections import Counter


app = Flask(__name__)


bp = Blueprint(
    "decision_tree", __name__, url_prefix="/decision_tree", template_folder="templates"
)
data = {
    "Troi": [
        "Rain",
        "Rain",
        "Overcast",
        "Sunny",
        "Sunny",
        "Sunny",
        "Overcast",
        "Rain",
        "Rain",
        "Sunny",
        "Rain",
        "Overcast",
        "Overcast",
        "Sunny",
    ],
    "Nhiet do": [
        "Hot",
        "Hot",
        "Hot",
        "Mild",
        "Cool",
        "Cool",
        "Cool",
        "Mild",
        "Cool",
        "Mild",
        "Mild",
        "Mild",
        "Hot",
        "Mild",
    ],
    "Do am": [
        "High",
        "High",
        "High",
        "High",
        "Normal",
        "Normal",
        "Normal",
        "High",
        "Normal",
        "Normal",
        "Normal",
        "High",
        "Normal",
        "High",
    ],
    "Gio": [
        "Weak",
        "Strong",
        "Weak",
        "Weak",
        "Weak",
        "Strong",
        "Strong",
        "Weak",
        "Weak",
        "Weak",
        "Strong",
        "Strong",
        "Weak",
        "Strong",
    ],
    "Choi Golf": [
        "No",
        "No",
        "Yes",
        "Yes",
        "Yes",
        "No",
        "Yes",
        "No",
        "Yes",
        "Yes",
        "Yes",
        "Yes",
        "Yes",
        "No",
    ],
}
df = pd.DataFrame(data)


# Tính Entropy
def caculate_entropy(df, target):
    counts = Counter(df[target])
    total_samples = len(df)
    entropy = 0
    for label in counts:
        p = counts[label] / total_samples
        entropy -= p * np.log2(p) if p > 0 else 0
    return entropy


# Tinh Information Gain
def caculate_information_gain(df, feature, target, initial_entropy):
    feature_values = df[feature].unique()
    weighted_entropy = 0
    for value in feature_values:
        subset = df[df[feature] == value]
        p = len(subset) / len(df)
        weighted_entropy += p * caculate_entropy(subset, target)
    return initial_entropy - weighted_entropy


# Tìm thuộc tính tốt nhất
def find_best_feature(df, features, target):
    initial_entropy = caculate_entropy(df, target)
    max_gain = -1
    best_feature = None
    for feature in features:
        gain = caculate_information_gain(df, feature, target, initial_entropy)
        if gain > max_gain:
            max_gain = gain
            best_feature = feature
    return best_feature, max_gain


class ID3DecisionTree:
    def __init__(self, df, features, target):
        self.df = df
        self.features = features
        self.target = target
        self.tree = self.build_tree(self.df, self.features)

    def build_tree(self, df, features):
        # B1: các mẫu là 1 lớp
        if len(df[self.target].unique()) == 1:
            return df[self.target].iloc[0]
        if not features:
            return Counter(df[self.target]).most_common(1)[0][0]
        # B2:
        # Tìm thuộc tính tốt nhất
        best_feature, _ = find_best_feature(df, features, self.target)
        tree = {best_feature: {}}
        remaining_features = [f for f in features if f != best_feature]
        # Tạo cây cho mõi giá trị có thuộc tính tốt nhất
        for value in df[best_feature].unique():
            subset = df[df[best_feature] == value].copy()
            if not subset.empty:
                tree[best_feature][value] = self.build_tree(subset, remaining_features)
            else:
                tree[best_feature][value] = Counter(df[self.target]).most_common(1)[0][
                    0
                ]
        return tree


def predict_with_tree(tree, sample):
    if not isinstance(tree, dict):
        return tree
    feature_name = list(tree.keys())[0]
    sub_tree = tree[feature_name]
    feature_value = sample[feature_name]
    if feature_value in sub_tree:
        return predict_with_tree(sub_tree[feature_value], sample)
    else:
        return "No"  # Default value if not found


# ---
# Thực thi
features = ["Troi", "Nhiet do", "Do am", "Gio"]
target = "Choi Golf"
# Bước 1: Tinh Information Gian:
inittial_entropy = caculate_entropy(df, target)
print("Entropy goc: {inittial_entropy}")
# Bước 2: Xây dựng cây và in ra tree
id3_tree = ID3DecisionTree(df, features, target)
print("Cau truc cay quy dinh (ID3): ")
print(id3_tree.tree)


# -----
# Tạo thư mục 'static' nếu chưa tồn tại
STATIC_FOLDER = os.path.join(os.getcwd(), "static")
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)


# Hàm để vẽ cây từ cấu trúc dictionary
def draw_decision_tree(tree, filename="decision_tree.png"):
    dot = graphviz.Digraph(comment="Decision Tree", format="png")
    node_counter = 0

    def add_nodes_edges(sub_tree, parent_id=None, edge_label=""):
        nonlocal node_counter
        current_id = str(node_counter)
        node_counter += 1

        if isinstance(sub_tree, dict):
            # Nút nhánh (branch node)
            feature_name = list(sub_tree.keys())[0]
            label = f"Feature: {feature_name}"
            dot.node(
                current_id,
                label,
                shape="ellipse",
                style="filled",
                fillcolor="lightblue",
            )

            # Thêm các cạnh va nut con
            for value, child_tree in sub_tree[feature_name].items():
                child_id = add_nodes_edges(child_tree, current_id, str(value))
                dot.edge(current_id, child_id, label=str(value))
        else:
            # Nút lá (leaf node)
            label = f"Result: {sub_tree}"
            dot.node(
                current_id, label, shape="box", style="filled", fillcolor="lightgreen"
            )

        if parent_id:
            return current_id
        else:
            return current_id

    add_nodes_edges(tree)
    filename = os.path.join(STATIC_FOLDER, filename)
    dot.render(filename=os.path.splitext(filename)[0], cleanup=True)
    print(f"Decision tree saved to: {filename}")


draw_decision_tree(id3_tree.tree)


@bp.route("/", methods=["GET", "POST"])
def index():
    tree_image_url = url_for("static", filename="decision_tree.png")
    if request.method == "POST":

        input_data = {
            "Troi": request.form.get("Troi"),
            "Nhietdo": request.form.get("Nhietdo"),
            "Doam": request.form.get("Doam"),
            "Gio": request.form.get("Gio"),
        }

        prediction = predict_with_tree(id3_tree.tree, input_data)
        print("ket qua du doan: + {prediction}")
        # Trả về trang result.html với kết quả dự đoán
        return render_template(
            "result_d.html",
            prediction=prediction,
            input_data=input_data,
            tree_image_url=tree_image_url,
            table=df.to_html(index=False),
        )
    return render_template(
        "index_d.html", tree_image_url=tree_image_url, table=df.to_html(index=False)
    )


app.register_blueprint(bp)
if __name__ == "__main__":
    app.run(debug=True)
