from flask import Flask, Blueprint, render_template, request
import numpy as np

app = Flask(__name__)

bp = Blueprint(
    'k_means',
    __name__,
    url_prefix='/k_means',
    template_folder='templates'
)

def generate_matrix(rows, cols):
    if cols < rows:
        raise ValueError("Số điểm phải lớn hơn hoặc bằng số cụm")

    matrix = np.zeros((rows, cols), dtype=int)

    chosen_cols = np.random.choice(cols, rows, replace=False)  
    for row in range(rows):
        matrix[row, chosen_cols[row]] = 1

    for col in range(cols):
        if 1 not in matrix[:, col]:
            row = np.random.randint(0, rows)
            matrix[row, col] = 1
    return matrix

def kmeans(points, k):
    points = np.array(points)
    n = len(points)

    rows = k
    cols = n
    
    ma_tran_phan_hoach = generate_matrix(rows, cols)

    flag = True
    count = 0

    final_dict = {}

    while flag:
        count += 1
        danh_sach_trong_tam = []
        for i in range(k):
            idxs = np.where(ma_tran_phan_hoach[i] == 1)[0]
            diem_trong_cum = points[idxs]
            idx_diem_dict = {f'điểm {idx+1}': tuple(diem_trong_cum[j]) for j, idx in enumerate(idxs)}
            trong_tam = diem_trong_cum.mean(axis=0)
            danh_sach_trong_tam.append(trong_tam)
            final_dict[f"Cụm_{i+1}"] = {"điểm trong cụm": idx_diem_dict, "trọng tâm": list(trong_tam)}
        
        centroids = np.array(danh_sach_trong_tam)
        distances = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)

        ma_tran_phan_hoach_tinh_lai = np.zeros((rows, cols), dtype=int)
        for idx, label in enumerate(labels):
            ma_tran_phan_hoach_tinh_lai[label, idx] = 1
        
        if np.array_equal(ma_tran_phan_hoach, ma_tran_phan_hoach_tinh_lai):
            flag = False
        else:
            ma_tran_phan_hoach = ma_tran_phan_hoach_tinh_lai
        
    return final_dict, count

@bp.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        diem_input = request.form.get("diem")
        cum_input = request.form.get("cum")
        points = diem_input.replace("(", "").replace(")", "").split(",")
        points = [list(map(float, point.strip().split())) for point in points if point.strip()]
        k = int(cum_input)
        final_dict, count = kmeans(points, k)
        return render_template("result_k.html", result=final_dict, count=count, diem=diem_input, cum=cum_input)

    return render_template("index_k.html")

app.register_blueprint(bp)

if __name__ == "__main__":
    app.run(debug=True)
