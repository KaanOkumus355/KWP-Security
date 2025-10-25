# -------------------------------------------------
# train_spam.py
# -------------------------------------------------
import pandas as pd
from pathlib import Path
import re
import nltk
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


CSV_PATH   = Path(__file__).parent / "data" / "spam.csv"
MODEL_DIR  = Path(__file__).parent / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 1️⃣ READ CSV ----------
try:
    df = pd.read_csv(
        CSV_PATH,
        header=None,
        usecols=[0, 1],
        names=["label", "text"],
        encoding="utf-8",
        engine="python"
    )
except UnicodeDecodeError:
    df = pd.read_csv(CSV_PATH, encoding="latin-1")

print(f"✅ Loaded {len(df)} rows from {CSV_PATH}")
print("📋 Original columns →", df.columns.tolist())

df = df.dropna(subset=[df.columns[0], df.columns[1]]).reset_index(drop=True)

# ---------- 2️⃣ LABEL → integer ----------
orig_label_col = df.columns[0]          # keep original name (e.g. v1)
df["orig_label"] = df[orig_label_col]  # backup copy
label_map = {"ham": 0, "legitimate": 0, "spam": 1, "phishing": 1}
df["label"] = df[orig_label_col].map(label_map).astype(int)

# ---------- 3️⃣ TEXT CLEANING ----------
# **No stop‑word removal, no character stripping – only get rid of the replacement char**
def clean_text(raw: str) -> str:
    """
    * Replace the Unicode replacement character � (U+FFFD) with a space.
    * Collapse consecutive whitespace to a single space.
    * Keep the text exactly as‑is (including URLs, punctuation, numbers, $ symbols, etc.).
    """
    txt = raw.replace("\uFFFD", " ")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

text_col = df.columns[1]                 # keep original name (e.g. v2)
df["clean"] = df[text_col].apply(clean_text)

# ---------- 4️⃣ TRAIN / TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label"],
    test_size=0.20,
    stratify=df["label"],
    random_state=42
)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=8000,          # give it a little more room
    min_df=2,
    sublinear_tf=True,
    stop_words=None             # ← **no stop‑word removal**
)


X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# ---------- 6️⃣ LOGISTIC REGRESSION ----------
clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="lbfgs"
)
clf.fit(X_train_vec, y_train)

# ---------- 7️⃣ EVALUATE ----------
y_prob = clf.predict_proba(X_test_vec)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

print("\n=== Classification report ===")
print(classification_report(y_test, y_pred,
                            target_names=["ham", "phishing"]))
print("ROC‑AUC :", roc_auc_score(y_test, y_prob))

# ---------- 8️⃣ SAVE ----------
joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")
joblib.dump(clf,       MODEL_DIR / "logreg_spam.pkl")
print(f"\n✅ Model artefacts written to {MODEL_DIR}")
