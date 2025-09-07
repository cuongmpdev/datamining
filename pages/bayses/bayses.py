from flask import Blueprint, render_template, request, flash
import csv
import io

from .nb import NaiveBayesCategorical

bp = Blueprint("bayses", __name__, url_prefix='/bayses', template_folder="templates")

def parse_csv(raw: str) -> list[dict]:
    f = io.StringIO(raw.strip())
    reader = csv.DictReader(f)
    rows = [ {k.strip(): (v.strip() if v is not None else "") for k,v in row.items()} for row in reader ]
    return rows

def parse_kv(sample_text: str) -> dict:
    """
    Chuỗi dạng: Thời tiết=Nắng; Nhiệt độ=Nóng; Độ ẩm=Cao; Gió=Yếu
    hoặc key=value; key=value
    """
    sample = {}
    for pair in sample_text.split(";"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            sample[k.strip()] = v.strip()
    return sample

@bp.route("/", methods=["GET"])
def form():
    example_csv = (
        "Thời tiết,Nhiệt độ,Độ ẩm,Gió,Đi chơi?\n"
        "Nắng,Nóng,Cao,Yếu,No\n"
        "Nắng,Nóng,Cao,Mạnh,No\n"
        "U ám,Nóng,Cao,Mạnh,Yes\n"
        "Mưa,Mát,Cao,Yếu,Yes\n"
        "Mưa,Lạnh,Cao,Mạnh,No\n"
        "Mưa,Lạnh,Bình thường,Mạnh,No\n"
        "U ám,Lạnh,Bình thường,Yếu,Yes\n"
        "Nắng,Mát,Cao,Yếu,No\n"
        "Nắng,Lạnh,Bình thường,Yếu,Yes\n"
    )
    return render_template(
        "bayses/form.html",
        example_csv=example_csv
    )

@bp.route("/run", methods=["POST"])
def run_bayes():
    try:
        training_csv = request.form.get("training_csv", "")
        target_col = request.form.get("target_col", "").strip()
        sample_text = request.form.get("sample_kv", "")
        alpha = float(request.form.get("alpha", "1.0"))

        if not training_csv or not target_col or not sample_text:
            flash("Vui lòng nhập đầy đủ dữ liệu.")
            return render_template("bayses/form.html", example_csv="")

        rows = parse_csv(training_csv)
        sample = parse_kv(sample_text)

        # xác thực khóa mẫu khớp header (trừ cột target)
        headers = list(rows[0].keys())
        features = [h for h in headers if h != target_col]
        missing = [f for f in features if f not in sample]
        if missing:
            flash(f"Thiếu giá trị cho các thuộc tính: {', '.join(missing)}")
            return render_template("bayses/form.html", example_csv=training_csv)

        nb = NaiveBayesCategorical(alpha=alpha)
        nb.fit(rows, target_col=target_col)
        proba = nb.predict_proba(sample)
        explain = nb.explain(sample)

        # sắp xếp theo xác suất giảm dần
        proba_sorted = sorted(proba.items(), key=lambda x: x[1], reverse=True)
        pred_label, pred_score = proba_sorted[0]

        return render_template(
            "bayses/result.html",
            target_col=target_col,
            features=features,
            sample=sample,
            pred_label=pred_label,
            pred_score=pred_score,
            proba=proba_sorted,
            explain=explain,
            alpha=alpha,
            n_rows=len(rows)
        )
    except Exception as e:
        flash(f"Lỗi: {e}")
        return render_template("bayses/form.html", example_csv="")