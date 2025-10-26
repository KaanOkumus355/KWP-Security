# -------------------------------------------------
# train_spam.py   – train + save the enhanced pipeline
# -------------------------------------------------
import re
import pandas as pd
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion

# Import all custom transformers (the file you just created)
from feature_transformers import (
    UrlFlagTransformer,
    KeywordFlagTransformer,
    DomainWhitelistTransformer,
    FirstPersonPronounTransformer
)

# ---------- CONFIG ----------
CSV_PATH   = Path(__file__).parent / "data" / "spam.csv"
MODEL_PATH = Path(__file__).parent / "model" / "spam_detector.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------- 0️⃣ READ ----------
try:
    df = pd.read_csv(
        CSV_PATH,
        header=None,
        usecols=[0, 1],
        names=["label", "text"],
        encoding="utf-8",
        engine="python")
except UnicodeDecodeError:
    df = pd.read_csv(
        CSV_PATH,
        header=None,
        usecols=[0, 1],
        names=["label", "text"],
        encoding="latin-1",
        engine="python")

df = df.dropna(subset=[df.columns[0], df.columns[1]]).reset_index(drop=True)

# ---------- 1️⃣ LABEL ----------
orig_label_col = df.columns[0]
df["orig_label"] = df[orig_label_col]          # backup copy (optional)

label_map = {"ham": 0, "legitimate": 0, "spam": 1, "phishing": 1}
df["label"] = df[orig_label_col].map(label_map)

# Drop rows that could not be mapped (e.g., stray NaN labels)
unknown = df["label"].isna()
if unknown.any():
    print("⚠️  Dropping rows with unknown label values:",
          df.loc[unknown, orig_label_col].unique())
    df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

# ---------- 2️⃣ CLEAN ----------
def clean_text(raw: str) -> str:
    txt = raw.lower()
    txt = raw.replace("\uFFFD", " ")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

text_col = df.columns[1]
df["clean"] = df[text_col].apply(clean_text)

# ---------- 3️⃣ TRAIN / TEST ----------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label"],
    test_size=0.20,
    stratify=df["label"],
    random_state=42)

# ---------- 4️⃣ PIPELINE ----------
pipeline = Pipeline([
    ("features", FeatureUnion([
        # ---- WORD‑LEVEL TF‑IDF (keep everything) ----
        ("tfidf_word", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=None,      # no hard cap
            min_df=1,
            sublinear_tf=True,
            stop_words=None
        )),
        # ---- CHARACTER‑LEVEL TF‑IDF (captures obfuscation) ----
        ("tfidf_char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
            max_features=5000
        )),
        ("url_flag", UrlFlagTransformer()),
        ("keyword_flag", KeywordFlagTransformer()),
        ("whitelist", DomainWhitelistTransformer()),       # NEW
        ("first_person", FirstPersonPronounTransformer()) # NEW (optional)
    ])),
    ("clf", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs"
    ))
])

# ---------- 5️⃣ FIT ----------
pipeline.fit(X_train, y_train)

# ---------- 6️⃣ EVALUATE ----------
y_prob = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

print("\n=== Classification report ===")
print(classification_report(y_test, y_pred,
                            target_names=["ham", "phishing"]))
print("ROC‑AUC :", roc_auc_score(y_test, y_prob))

# ---------- 7️⃣ SAVE ----------
joblib.dump(pipeline, MODEL_PATH)
print(f"\n✅ Pipeline saved to {MODEL_PATH}")
