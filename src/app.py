import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ===============================
# 1️⃣ LOAD MODEL + SCALER + THRESHOLD (an toàn đường dẫn)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))  # cha của src/
# Nếu app.py ở root thì PROJECT_ROOT == BASE_DIR; nếu ở src/ thì PROJECT_ROOT là root

def load_artifact(*relative_parts):
    """Tìm file theo 2 khả năng: tại BASE_DIR và tại PROJECT_ROOT."""
    candidate1 = os.path.join(BASE_DIR, *relative_parts)
    candidate2 = os.path.join(PROJECT_ROOT, *relative_parts)
    path = candidate1 if os.path.exists(candidate1) else candidate2
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file: {relative_parts} (đã thử {candidate1} và {candidate2})")
    return joblib.load(path)

scaler = load_artifact("models", "models_scaler_xgb1.pkl")
model = load_artifact("models", "models_xgb_trained_model1.pkl")
best_threshold = float(load_artifact("models", "models_best_threshold_xgb1.pkl"))

# ===============================
# 2️⃣ FLASK APP
# ===============================
# Trỏ đúng thư mục templates/static ở root hoặc src
templates_dir = os.path.join(BASE_DIR, "templates") if os.path.exists(os.path.join(BASE_DIR, "templates")) else os.path.join(PROJECT_ROOT, "templates")
static_dir = os.path.join(BASE_DIR, "static") if os.path.exists(os.path.join(BASE_DIR, "static")) else os.path.join(PROJECT_ROOT, "static")

app = Flask(__name__, template_folder=templates_dir, static_folder=static_dir)
CORS(app)

# ===============================
# 3️⃣ DANH SÁCH CỘT ONE-HOT
# ===============================
LANG_LIST = [
    "english", "spanish", "french", "german", "chinese",
    "japanese", "korean", "hindi"
]

GENRE_LIST = [
    "action", "comedy", "drama", "horror",
    "scifi", "romance", "thriller", "animation"
]

ALL_FEATURES = (
    ["start_year", "runtime_minutes", "num_votes", "num_votes_log"]
    + GENRE_LIST
    + LANG_LIST
)

# ===============================
# 4️⃣ HÀM XỬ LÝ INPUT -> VECTOR (kèm kiểm tra)
# ===============================
def preprocess_input(year, runtime, language=None, genre=None):
    try:
        year = int(year)
        runtime = int(runtime)
    except (TypeError, ValueError):
        raise ValueError("year và runtime phải là số nguyên hợp lệ")

    runtime_adj = runtime
    num_votes_adj = 2000  # votes giả lập

    if runtime > 60:
        runtime_adj *= 0.8

    if language not in LANG_LIST:
        num_votes_adj *= 0.5

    if year < 1950:
        num_votes_adj *= 0.5
        runtime_adj *= 0.2

    numeric = np.array([
        year,
        runtime_adj,
        num_votes_adj,
        np.log1p(num_votes_adj)
    ], dtype=float).reshape(1, -1)

    numeric_scaled = scaler.transform(numeric)

    onehot = np.zeros(len(GENRE_LIST) + len(LANG_LIST), dtype=float)
    if genre in GENRE_LIST:
        onehot[GENRE_LIST.index(genre)] = 1
    if language in LANG_LIST:
        onehot[len(GENRE_LIST) + LANG_LIST.index(language)] = 1

    X = np.concatenate([numeric_scaled[0], onehot]).reshape(1, -1)
    return X

# ===============================
# 5️⃣ ROUTE DỰ BÁO (kiểm tra input + thông báo lỗi rõ ràng)
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    # Hỗ trợ cả JSON và form-encoded
    data = request.get_json(silent=True) or request.form.to_dict()

    title = data.get("title")
    year = data.get("year")
    runtime = data.get("runtime")
    language = data.get("language")
    genre = data.get("genre")

    if year in (None, "") or runtime in (None, ""):
        return jsonify({"error": "Missing year or runtime"}), 400

    try:
        X = preprocess_input(year, runtime, language, genre)
        proba = float(model.predict_proba(X)[0][1])
        pred = int(proba >= best_threshold)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    verdict = "Phim hay" if pred == 1 else "Phim dở"
    return jsonify({
        "title": title,
        "prediction": verdict,
        "score": round(proba * 100, 2)
    })

# ===============================
# 6️⃣ ROUTE HOME
# ===============================
@app.route("/")
def home_page():
    return render_template("index.html")

# (Tuỳ chọn) Healthcheck để kiểm tra nhanh
@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200

# ===============================
# 7️⃣ RUN APP
# ===============================
if __name__ == "__main__":
    # Cho Codespaces: mở cổng ra ngoài
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)