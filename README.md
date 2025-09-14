Data Mining Web — Tiếng Việt

Tổng quan
- Frontend: React + Tailwind (mẫu theo shadcn), có sidebar trái để chọn thuật toán.
- Backend: FastAPI triển khai 4 thuật toán từ đầu (không dùng thư viện ML):
  - Clustering: K-Means (mục tiêu SSE, khởi tạo k-means++)
  - Bayes: Naive Bayes (Gaussian và Multinomial)
  - Decision Tree: ID3 dùng Entropy/Information Gain, hỗ trợ ngưỡng cho thuộc tính số
  - Reduct: Rough Set QuickReduct (độ phụ thuộc γ và vùng dương tính)

Nếu học phần của bạn yêu cầu biến thể cụ thể (ví dụ C4.5 với Gain Ratio, hay công thức làm mịn Laplace khác), hãy cho biết để mình điều chỉnh.

Cấu trúc dự án
- backend: máy chủ FastAPI và mã thuật toán
- frontend: ứng dụng React, mỗi thuật toán có một trang cấu hình/ chạy riêng
- data: một số CSV ví dụ ở gốc repo (tham khảo); bản dùng trong UI nằm ở `frontend/public/data`

Yêu cầu hệ thống
- Python 3.9+ (khuyên dùng 3.10/3.11)
- Node.js 18+ và npm

Chạy Backend (FastAPI)
1) Tạo môi trường ảo và cài phụ thuộc
   - cd backend
   - python3 -m venv .venv && source .venv/bin/activate  (Windows: .venv\Scripts\activate)
   - pip install -r requirements.txt
2) Khởi động API server
   - uvicorn app:app --reload --port 8000
3) Kiểm tra nhanh
   - Mở http://localhost:8000/api/health → {"status":"ok"}

API endpoints
- GET /api/health
- POST /api/preview  (multipart: file)
- POST /api/kmeans  (multipart: file, k, [max_iter, tol, columns, random_state])
- POST /api/naive-bayes  (multipart: file, target, [variant=gaussian|multinomial, alpha, feature_columns])
- POST /api/decision-tree  (multipart: file, target, [max_depth, min_samples_split])
- POST /api/reduct  (multipart: file, decision, [conditional])

Chạy Frontend (React + Vite)
1) Cài phụ thuộc
   - cd frontend
   - npm install
2) Cấu hình API base (tuỳ chọn, mặc định http://localhost:8000)
   - Tạo file `.env` với dòng: `VITE_API_BASE=http://localhost:8000`
3) Chạy dev server
   - npm run dev (mặc định: http://localhost:5173)

Sử dụng giao diện
- Mỗi trang thuật toán cho phép:
  1) Tải lên CSV từ máy (nút “Chọn CSV”), hoặc
  2) Chọn nhanh dữ liệu mẫu phù hợp thuật toán (các nút ngay dưới ô chọn tệp).
- Quy ước chung: cột mục tiêu/nhãn mặc định là cột cuối cùng, bạn có thể đổi trong dropdown.
- Naive Bayes: khi chọn dữ liệu mẫu có chữ “multinomial” trong tên, trang sẽ tự chuyển sang biến thể Multinomial; ngược lại là Gaussian.

Dữ liệu CSV mẫu (hiển thị theo từng trang)
- Decision Tree: `play_tennis.csv`, `decision_tree_loan.csv`, `decision_tree_mushroom.csv`
- K-Means: `kmeans_points.csv`, `kmeans_points_b.csv`, `kmeans_points_c.csv`
- Naive Bayes: `naive_bayes_gaussian.csv`, `naive_bayes_gaussian2.csv`, `naive_bayes_multinomial.csv`, `naive_bayes_multinomial2.csv`
- Reduct: `play_tennis.csv`, `decision_tree_loan.csv`, `decision_tree_mushroom.csv`

Kỳ vọng CSV
- Tệp CSV mã hoá UTF-8, có hàng tiêu đề (header) ở dòng đầu tiên.
- Endpoint preview sẽ suy luận kiểu cột (numeric vs categorical). Khi ép kiểu số, giá trị thiếu sẽ mặc định thành 0.
- Lưu ý cho Multinomial NB: các đặc trưng phải không âm (count/ tần suất).

Chi tiết thuật toán (tóm tắt)
- K-Means: Tối thiểu SSE; khởi tạo k-means++ để hội tụ tốt hơn; dừng khi dịch chuyển tâm ≤ `tol` hoặc chạm `max_iter`.
- Naive Bayes:
  - Gaussian: Tính mean/variance theo lớp cho mỗi đặc trưng số; cộng log-likelihood và log-prior.
  - Multinomial: Dựa trên đếm, có làm mịn Laplace α; yêu cầu đặc trưng không âm.
- Decision Tree (ID3): Dùng entropy & information gain. Với cột số, thử các ngưỡng giữa giá trị khi nhãn thay đổi. Dự đoán bằng duyệt cây.
- Reduct (Rough Set): QuickReduct tham lam, tối đa hoá γ_R(D) đến khi đạt γ_C(D); dùng positive region và kiểm tra dư thừa.

Ghi chú về shadcn/ui
- UI dùng Tailwind với các component nhỏ (Button, Input, Card) theo phong cách shadcn.
- Nếu muốn thay bằng shadcn/ui chính thức:
  - npx shadcn@latest init
  - npx shadcn@latest add button input label card select ...
  - Thay thế các import ở `src/components/ui`

Mẹo & xử lý sự cố
- 404/Network khi gọi API từ frontend: kiểm tra backend đã chạy ở cổng 8000 và biến `VITE_API_BASE` trỏ đúng; bật CORS đã được cấu hình cho phát triển.
- CSV không hiện đúng cột số: xem endpoint preview để biết kiểu suy luận; đảm bảo dữ liệu là số (dấu phẩy thập phân dùng dấu `.`).
- Multinomial NB báo lỗi/cho kết quả lạ: kiểm tra dữ liệu không âm; chọn đúng biến thể Multinomial.
- Node/PNPM lỗi build: đảm bảo Node 18+; chạy `npm install` lại.

Tùy chỉnh tiếp theo (tuỳ chọn)
- Chuyển tiêu chí cây sang Gain Ratio (C4.5) nếu học phần yêu cầu.
- Thêm trực quan hoá K-Means và confusion matrix.
- Thêm tách train/test có stratified và các thước đo.
- Hỗ trợ phân cụm phân cấp nếu cần.
