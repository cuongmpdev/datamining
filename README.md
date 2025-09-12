Data Mining Web Toolkit (React + FastAPI)

Overview
- Frontend: React + Tailwind (shadcn-ready) with a left sidebar to select algorithms.
- Backend: FastAPI with endpoints implementing four algorithms from scratch:
  - Clustering: K-Means (SSE objective, k-means++ init)
  - Bayes: Naive Bayes (Gaussian and Multinomial)
  - Decision Tree: ID3 using Entropy/Information Gain, supports numeric thresholds
  - Reduct: Rough Set QuickReduct (dependency degree γ and positive region)

This aligns with typical course formulas in the attached PDFs. If your course requires specific variants (e.g., C4.5 with Gain Ratio for trees, or specific Laplace smoothing form), let me know and I will adapt.

Project Structure
- backend: FastAPI server and algorithms
- frontend: React app with sidebar and forms per algorithm
- data: sample CSV files for quick testing

Backend: Run Locally
1) Create a virtualenv and install deps
   - cd backend
   - python -m venv .venv && source .venv/bin/activate (or .venv\Scripts\activate on Windows)
   - pip install -r requirements.txt
2) Start API server
   - uvicorn app:app --reload --port 8000
3) API endpoints
   - GET /api/health
   - POST /api/preview (multipart: file)
   - POST /api/kmeans (multipart: file, k, [max_iter, tol, columns, random_state])
   - POST /api/naive-bayes (multipart: file, target, [variant=gaussian|multinomial, alpha, feature_columns])
   - POST /api/decision-tree (multipart: file, target, [max_depth, min_samples_split])
   - POST /api/reduct (multipart: file, decision, [conditional])

Frontend: Run Locally
1) Install deps
   - cd frontend
   - npm install
2) Start dev server
   - npm run dev (default: http://localhost:5173)
3) Configure API base (optional)
   - Create .env with VITE_API_BASE=http://localhost:8000

Notes on shadcn/ui
- The UI uses Tailwind with small utility components (Button, Input, Card) compatible with shadcn patterns.
- If you want to replace them with official shadcn/ui components:
  - npx shadcn@latest init
  - npx shadcn@latest add button input label card select ...
  - Replace usages under src/components/ui

CSV Expectations
- Upload a UTF-8 CSV with headers in the first row.
- Preview endpoint infers column types (numeric vs categorical). Missing values default to 0 in numeric casting.

Algorithm Details (High Level)
- K-Means: Minimizes SSE. Initialization via k-means++ to improve convergence. Stops when centroid shift ≤ tol or max_iter.
- Naive Bayes:
  - Gaussian: Per-class mean/variance for each numeric feature; log-likelihoods summed with log-priors.
  - Multinomial: Count-based with Laplace smoothing α; requires non-negative features.
- Decision Tree (ID3): Uses entropy and information gain. For numeric features, selects best threshold between unique sorted values where class changes. Predicts by traversing the tree.
- Reduct (Rough Set): QuickReduct greedy selection maximizing dependency degree γ_R(D) until reaching γ_C(D). Computes positive region via indiscernibility classes; includes redundancy check.

Next Adjustments (if needed)
- Switch tree criterion to Gain Ratio (C4.5) if your course requires.
- Add k-means visualization and confusion matrices.
- Add stratified train/test split and metrics.
- Support hierarchical clustering if covered in your slides.
Sample CSVs
- `data/kmeans_points.csv`: 2D numeric points with 3 clear clusters. Use with K-Means, select both `x` and `y`.
- `data/naive_bayes_gaussian.csv`: Numeric features (`height`,`weight`) with class `label` in the last column. Use Naive Bayes (Gaussian), set target to `label`.
- `data/naive_bayes_multinomial.csv`: Count features (`free,offer,click,meeting,project`) with target `label` (spam/ham). Use Naive Bayes (Multinomial), target `label`.
- `data/play_tennis.csv`: Categorical attributes with decision `Play` in the last column. Use for Decision Tree (ID3) and Reduct.

Tip: In the UI pages, the target defaults to the last column. You can change it from the dropdown.
