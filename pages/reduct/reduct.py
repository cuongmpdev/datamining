from flask import Flask, Blueprint, render_template, request
import pandas as pd
import itertools

app = Flask(__name__)

bp = Blueprint(
    'reduct',
    __name__,
    url_prefix='/reduct',
    template_folder='templates'
)

data = {
    "O": ["O1","O2","O3","O4","O5","O6","O7","O8"],
    "Troi": ["Trong","May","May","Trong","May","May","May","Trong"],
    "Gio": ["Bac","Nam","Bac","Bac","Bac","Bac","Nam","Nam"],
    "Apsuat": ["Cao","Cao","TB","Thap","Thap","Cao","Thap","Cao"],
    "Ketqua": ["Kmua","Mua","Mua","Kmua","Mua","Mua","Kmua","Kmua"]
}
df = pd.DataFrame(data)
attr_names = ["Troi","Gio","Apsuat"]

def get_equivalence_classes(df, attributes):
    if not attributes:
        return [[obj] for obj in df["O"]]
    groups = df.groupby(attributes, dropna=False)
    return [list(group["O"]) for _, group in groups]

def lower_upper_approximation(df, X, B):
    eq_classes = get_equivalence_classes(df, B)
    lower = [obj for cls in eq_classes if set(cls).issubset(X) for obj in cls]
    upper = [obj for cls in eq_classes if set(cls).intersection(X) for obj in cls]
    alpha = len(lower) / len(upper) if len(upper) > 0 else 0
    return lower, upper, alpha

def is_reduct(attrs):
    groups = {}
    for row in df.itertuples(index=False):
        key = tuple(getattr(row, a) for a in attrs)
        groups.setdefault(key, []).append(row)
    for g in groups.values():
        decs = set(r.Ketqua for r in g)
        if len(decs) > 1:
            return False
    return True

def find_reducts_and_core():
    reducts = []
    for r in range(1, len(attr_names)+1):
        for combo in itertools.combinations(attr_names, r):
            if is_reduct(combo):
                # kiểm tra tối tiểu
                minimal = True
                for k in range(1, len(combo)):
                    for sub in itertools.combinations(combo, k):
                        if is_reduct(sub):
                            minimal = False
                            break
                    if not minimal:
                        break
                if minimal:
                    reducts.append(combo)
    core = set(reducts[0]) if reducts else set()
    for r in reducts[1:]:
        core &= set(r)
    return reducts, core

@bp.route("/", methods=["GET", "POST"])
def index():
    reducts, core = find_reducts_and_core()

    if request.method == "POST":
        X_input = request.form.get("X")
        B_input = request.form.get("B")

        X = set(x.strip() for x in X_input.split(",") if x.strip())
        B = [b.strip() for b in B_input.split(",") if b.strip()]

        lower, upper, alpha = lower_upper_approximation(df, X, B)

        return render_template(
            "result.html",
            lower=lower, upper=upper, alpha=alpha,
            B=B, X=X, table=df.to_html(index=False),
            reducts=reducts, core=core
        )

    return render_template(
        "index.html",
        table=df.to_html(index=False),
        reducts=reducts, core=core
    )

app.register_blueprint(bp)

if __name__ == "__main__":
    app.run(debug=True)
