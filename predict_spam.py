# -------------------------------------------------
# predict_spam.py   – load the model, vectorise, and predict
# -------------------------------------------------
import re
import joblib
from pathlib import Path

# Import the custom transformers so un‑pickling can locate them.
# They are not used directly in this script, but Python needs the definitions.
from feature_transformers import (               # noqa: F401
    UrlFlagTransformer,
    KeywordFlagTransformer,
    DomainWhitelistTransformer,
    FirstPersonPronounTransformer
)

# ---------- 0️⃣ PATH ----------
MODEL_PATH = Path(__file__).parent / "model" / "spam_detector.pkl"
pipeline   = joblib.load(MODEL_PATH)   # loads the full pipeline (features + classifier)

# ---------- 1️⃣ CLEANING (identical to training) ----------
def clean_text(raw: str) -> str:
    txt = raw.replace("\uFFFD", " ")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

# ---------- 2️⃣ EXAMPLES ----------
EXAMPLES = [
   
    "ATTENTION, youR BANK ACCOUNT IS BEING HACKED please log in at https://santandr.com"
]
# ---------- 3️⃣ VECTORISE ONLY THE FEATURES ----------
# The pipeline ends with a classifier, so we need the FeatureUnion that holds the actual transforms.
features_union = pipeline.named_steps["features"]
X_matrix = features_union.transform(EXAMPLES)   # (n_samples, n_features)

print("\n🔢 Matrix shape :", X_matrix.shape)
print("🧩 Non‑zero entries per sample :", X_matrix.getnnz(axis=1))

# ---------- 4️⃣ QUICK LOOK AT FEATURES FOR THE LAST SAMPLE ----------
# Compose the complete feature‑name list in the same order as FeatureUnion concatenates them.
word_features   = features_union.transformer_list[0][1].get_feature_names_out().tolist()
char_features   = features_union.transformer_list[1][1].get_feature_names_out().tolist()
url_feature     = ["URL_FLAG"]
keyword_features = features_union.transformer_list[3][1].keywords
whitelist_feat  = ["WHITELIST"]
first_person    = ["FIRST_PERSON"]
feature_names = (word_features + char_features + url_feature +
                 keyword_features + whitelist_feat + first_person)

# Show a few non‑zero entries of the *last* example (the Amazon one)
last_vec = X_matrix[-1]                     # last row
indices = last_vec.indices[:12]             # first 12 non‑zero columns
values  = last_vec.data[:12]

print("\n🧩 Amazon‑example – first 12 non-zero features:")
for idx, val in zip(indices, values):
    print(f"   {feature_names[idx]:<35} → {val:.4f}")

# ---------- 5️⃣ PREDICTION (full pipeline) ----------
proba = pipeline.predict_proba(EXAMPLES)[:, 1]   # phishing probability (class=1)
pred  = (proba >= 0.5).astype(int)              # binary label

print("\n🔎 Prediction results")
for txt, p, lab in zip(EXAMPLES, proba, pred):
    label = "phishing" if lab else "ham"
    print(f"{label:<9} ({p:.2%}) → {txt[:70]}...")
