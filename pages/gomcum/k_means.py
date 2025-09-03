import numpy as np

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
    print(ma_tran_phan_hoach)

    flag = True
    count = 0

    final_dict = {}

    while flag:
        count += 1
        danh_sach_trong_tam = []
        for i in range(k):
            idxs = np.where(ma_tran_phan_hoach[i] == 1)[0]
            diem_trong_cum = points[idxs]
            idx_diem_dict = {f'điểm A{idx}': tuple(diem_trong_cum[j]) for j, idx in enumerate(idxs)}
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
    
    print(f"Thuật toán k-means đã hội tụ sau {count} lần lặp.")
    print(f"Kết quả phân cụm cuối cùng: {final_dict}")

if __name__ == "__main__":
    n = int(input("Nhập số điểm: "))
    points = []
    for i in range(n):
        x, y = map(float, input(f"Nhập toạ độ điểm A{i+1} (x y): ").split())
        points.append([x, y])
    
    k = int(input("Nhập số cụm k: "))
    
    kmeans(points, k)
