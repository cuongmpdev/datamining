import numpy as np

def generate_matrix(rows, cols):
    if cols < rows:
        raise ValueError("Số điểm phải lớn hơn hoặc bằng số cụm")

    # Khởi tạo ma trận toàn 0
    matrix = np.zeros((rows, cols), dtype=int)

    # Bước 1: Đảm bảo mỗi hàng có ít nhất một số 1
    chosen_cols = np.random.choice(cols, rows, replace=False)  # chọn cột khác nhau cho mỗi hàng
    for row in range(rows):
        matrix[row, chosen_cols[row]] = 1

    # Bước 2: Với các cột còn lại chưa có 1, gán 1 vào một hàng ngẫu nhiên
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
    
    danh_sach_trong_tam = []
    for i in range(k):
        print(f"Cụm {i+1}:")
        print(ma_tran_phan_hoach[i] == 1)
        diem_trong_cum = points[ma_tran_phan_hoach[i] == 1]
        print(diem_trong_cum)
        trong_tam = diem_trong_cum.mean(axis=0)
        print(f"Trọng tâm cụm {i+1}: {trong_tam}")
        danh_sach_trong_tam.append(trong_tam)
    
    centroids = np.array(danh_sach_trong_tam)
    print(centroids)

    distances = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
    print("Khoảng cách giữa các điểm và các trọng tâm:")
    print(distances)

    labels = np.argmin(distances, axis=1)

    print("Nhãn của các điểm:")
    print(labels)
    # flag = True
    # i = 0
    # while flag:
    #     i += 1
    #     print(f"\nLần lặp thứ {i}:")
    #     # Bước 1: gán điểm vào cụm gần nhất
        
    

# --- Chương trình chính ---
if __name__ == "__main__":
    # n = int(input("Nhập số điểm: "))
    # points = []
    # for i in range(n):
    #     x, y = map(float, input(f"Nhập toạ độ điểm A{i+1} (x y): ").split())
    #     points.append([x, y])
    
    # k = int(input("Nhập số cụm k: "))
    
    # kmeans(points, k)
    kmeans([[1,9],[1,8],[3,6],[4,6],[8,2]], 3)