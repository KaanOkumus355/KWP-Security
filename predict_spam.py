# -------------------------------------------------
# predict_spam.py
# -------------------------------------------------
import joblib
import re
from pathlib import Path

# ------------------- CONFIG --------------------
MODEL_DIR = Path(__file__).parent / "model"
VEC_PATH  = MODEL_DIR / "tfidf_vectorizer.pkl"
CLS_PATH  = MODEL_DIR / "logreg_spam.pkl"

# ------------------- LOAD ----------------------
vectorizer = joblib.load(VEC_PATH)
clf        = joblib.load(CLS_PATH)

# ------------------- TEXT CLEANING ----------
def clean_text(raw: str) -> str:
    """
    Same cleaning logic used during training:
    * replace the Unicode replacement char � with a space
    * collapse whitespace
    * keep everything else exactly as it is
    """
    txt = raw.replace("\uFFFD", " ")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

# ------------------- PREDICT -----------------
def predict(message: str, thresh: float = 0.5) -> dict:
    cleaned = clean_text(message)
    vec = vectorizer.transform([cleaned])
    prob = clf.predict_proba(vec)[0, 1]          # probability of “phishing”
    label = "phishing" if prob >= thresh else "ham"
    return {"probability": prob, "label": label, "cleaned": cleaned}

# ------------------- DEMO --------------------
if __name__ == "__main__":
    example = "PRIVATE! Your 2003 Account Statement for 07808247860 shows 800 un-redeemed S. I. M. points. Call 08719899229 Identifier Code: 40411 Expires 06/11/04"
    out = predict(example)
    print("\n📝 Original :", example)
    print("🔄 Cleaned  :", out["cleaned"])
    print(f"📊 Phishing probability = {out['probability']:.2%}")
    print("🏷️ Predicted label    =", out["label"])
