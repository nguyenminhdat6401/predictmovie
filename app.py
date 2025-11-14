import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ===============================
# 1️⃣ LOAD MODEL + SCALER + THRESHOLD
# ===============================
scaler = joblib.load("models/models_scaler_xgb1.pkl")
model = joblib.load("models/models_xgb_trained_model1.pkl")
best_threshold = joblib.load("models/models_best_threshold_xgb1.pkl")

# ===============================
# 2️⃣ FLASK APP
# ===============================
app = Flask(__name__)
CORS(app)

# ===============================
# 3️⃣ DANH SÁCH CỘT ONE-HOT
# ===============================
LANG_LIST = [
    "english","spanish","french","german","chinese",
    "japanese","korean","hindi"
]

GENRE_LIST = [
    "action","comedy","drama","horror",
    "scifi","romance","thriller","animation"
]

ALL_FEATURES = (
    ["start_year", "runtime_minutes", "num_votes", "num_votes_log"]
    + GENRE_LIST
    + LANG_LIST
)

# ===============================
# 4️⃣ HÀM XỬ LÝ INPUT -> VECTOR
# ===============================
def preprocess_input(year, runtime, language=None, genre=None):
    year = int(year)
    runtime = int(runtime)

    runtime_adj = runtime
    num_votes_adj = 2000  # votes giả lập

    
    if runtime > 60:
        runtime_adj *= 0.8 


    if language not in LANG_LIST:
        num_votes_adj *= 0.5

    if year < 1950:
        num_votes_adj *= 0.5  # giảm votes để phản ánh nguy cơ
        runtime_adj *= 0.2    # giảm runtime tương đối
        
        
    # numeric features
    numeric = np.array([
        year,
        runtime_adj,
        num_votes_adj,
        np.log1p(num_votes_adj)
    ], dtype=float).reshape(1, -1)

    # scale numeric
    numeric_scaled = scaler.transform(numeric)

    # one-hot features
    onehot = np.zeros(len(GENRE_LIST) + len(LANG_LIST), dtype=float)
    if genre in GENRE_LIST:
        idx = GENRE_LIST.index(genre)
        onehot[idx] = 1
    if language in LANG_LIST:
        idx = len(GENRE_LIST) + LANG_LIST.index(language)
        onehot[idx] = 1

    # ghép vector
    X = np.concatenate([numeric_scaled[0], onehot]).reshape(1, -1)
    return X


# ===============================
# 5️⃣ ROUTE DỰ BÁO
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    title = data.get("title")       # chỉ dùng để hiển thị
    year = data.get("year")
    runtime = data.get("runtime")
    language = data.get("language") # chỉ dùng để hiển thị
    genre = data.get("genre")       # chỉ dùng để hiển thị

    # build input vector cho model
    X = preprocess_input(year, runtime, language, genre)

    # dự đoán xác suất
    proba = float(model.predict_proba(X)[0][1])
    pred = int(proba >= best_threshold)
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

# ===============================
# 7️⃣ RUN APP
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
