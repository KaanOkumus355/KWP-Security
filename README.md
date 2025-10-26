# Phishing Neural Network Using Logarithmic Regression

Based on a neural network:

*Logistic Regression:* Uses Logistic Function, Neural Network, Sigmoid


Need text segmentation to binary processing,

Three Steps:

*Binary Classifier*

*Gemini API integration*

*Launch UI*


Order of Neural Network:

Raw text  ──► preprocessing (clean, tokenize) ──► feature extraction ──► numeric vector x
x ──► logistic regression ──► probability p ──► decision (phishing vs ham)


# 📧 Phishing‑Detection Binary Classifier  

A lightweight, **explainable** model that decides whether a short text message is **phishing** (`spam`) or **legitimate** (`ham`).  
The pipeline combines classic **TF‑IDF** representations with a few **hand‑crafted binary features** (URL flag, domain whitelist, keyword counters, first‑person pronouns) and a **logistic‑regression** classifier.

---

## Table of Contents
1. [Problem statement](#problem-statement)  
2. [Data format](#data-format)  
3. [Mathematical model](#mathematical-model)  
4. [Feature engineering](#feature-engineering)  
   - [Word‑level TF‑IDF](#1-word‑level-tfidf)  
   - [Character‑level TF‑IDF](#2-character‑level-tfidf)  
   - [URL flag](#3-url‑flag)  
   - [Domain whitelist](#4-domain-whitelist)  
   - [Keyword counter](#5-keyword‑counter)  
   - [First‑person pronoun flag](#6-first‑person‑pronoun‑flag)  
5. [Training pipeline (code)](#training-pipeline-code)  
6. [Evaluation metrics](#evaluation-metrics)  
7. [Inference (code)](#inference-code)  
8. [How to run the code](#how-to-run-the-code)  
9. [Extending / customizing](#extending-customizing)  

---

## Problem statement <a name="problem-statement"></a>

Given a raw text message  

\[
\mathbf{s} = (c_1,c_2,\dots,c_L) \in \mathcal{V}^L
\]

(where \(\mathcal{V}\) is the character alphabet), predict a binary label  

\[
y \in \{0,1\},\qquad
y=1 \;\text{⇔ “phishing”},\; y=0 \;\text{⇔ “legitimate”}.
\]

The model is **linear** in a high‑dimensional sparse feature space \(\mathbf{x}\in\mathbb{R}^d\).

---

## Data format <a name="data-format"></a>

| Column | Description |
|--------|-------------|
| `label` | `"ham"` / `"spam"` (or `"legitimate"` / `"phishing"`). |
| `text`  | Raw message string, **no header row**. |
| **Location** | `data/spam.csv` (relative to the repository root). |

The script automatically maps the textual labels to integers (`0` = ham, `1` = spam).

---

## Mathematical model <a name="mathematical-model"></a>

### 1. Feature vector  

All preprocessing steps produce a **single sparse vector**  

\[
\mathbf{x}= \big[\,\mathbf{x}^{\text{word}};\; 
\mathbf{x}^{\text{char}};\;
x^{\text{url}};\;
\mathbf{x}^{\text{kw}};\;
x^{\text{wl}};\;
x^{\text{fp}}\big] \in \mathbb{R}^{d}
\]

where  

* \(\mathbf{x}^{\text{word}}\) – TF‑IDF on unigrams & bigrams,  
* \(\mathbf{x}^{\text{char}}\) – TF‑IDF on character 3‑5‑grams,  
* \(x^{\text{url}} \in \{0,1\}\) – presence of any URL,  
* \(\mathbf{x}^{\text{kw}}\) – counts of the **120+ phishing‑related keywords**,  
* \(x^{\text{wl}} \in \{0,1\}\) – domain belongs to a **whitelist** of trusted sites,  
* \(x^{\text{fp}} \in \{0,1\}\) – presence of a first‑person pronoun (`I`, `my`, `we`, …).

All components are **concatenated column‑wise**, producing a sparse matrix \(X\in\mathbb{R}^{n\times d}\) for the whole dataset.

### 2. Logistic regression  

The classifier learns a weight vector \(\mathbf{w}\) and bias \(b\) by solving the regularised log‑loss  

\[
\min_{\mathbf{w},b}\;
\frac{1}{n}\sum_{i=1}^{n}
\bigl[ -y_i\log \sigma(z_i) - (1-y_i)\log(1-\sigma(z_i))\bigr]
+ \lambda\|\mathbf{w}\|_2^{2},
\]

with  

\[
z_i = \mathbf{w}^{\!\top}\mathbf{x}_i + b,
\qquad
\sigma(z)=\frac{1}{1+e^{-z}} \;\text{(sigmoid)}.
\]

`scikit‑learn`’s `LogisticRegression` (solver **lbfgs**) does exactly this, using the **inverse regularisation strength** `C` (`C = 1/λ`).  
We set `class_weight='balanced'` so that the loss is automatically re‑weighted to counter any label imbalance.

The **output probability** for a new message \(\mathbf{s}\) is  

\[
\hat{p} = \sigma\bigl(\mathbf{w}^{\!\top}\mathbf{x}(\mathbf{s}) + b\bigr).
\]

A hard decision is derived by thresholding:  

\[
\hat{y}= \mathbf{1}\bigl(\hat{p}\ge\tau\bigr),
\]

with the default threshold \(\tau=0.5\) (you can tune it).

---

## Feature engineering <a name="feature-engineering"></a>

Below each feature is described **both conceptually and by the concrete Python implementation**.

### 1️⃣ Word‑level TF‑IDF <a name="1-word‑level-tfidf"></a>

| Concept | Code |
|---------|------|
| Unigrams + bigrams on the **cleaned** text. | ```python<br>TfidfVectorizer(ngram_range=(1,2), max_features=None, min_df=1,<br>                 sublinear_tf=True, stop_words=None)``` |
| TF‑IDF term weight: \(\displaystyle \text{tfidf}(t,d)=\text{tf}(t,d)\cdot\log\frac{N}{\text{df}(t)}\) (sub‑linear). | Handled internally by `scikit‑learn`. |

### 2️⃣ Character‑level TF‑IDF <a name="2-character‑level-tfidf"></a>

Captures **obfuscated words** and **URL fragments**.

| Concept | Code |
|---------|------|
| Character 3‑ to 5‑grams, word‑boundary aware (`analyzer='char_wb'`). | ```python<br>TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5),<br>                 min_df=1, sublinear_tf=True, max_features=5000)``` |
| Same TF‑IDF weighting as above. | `scikit‑learn` does the math. |

### 3️⃣ URL flag <a name="3-url‑flag"></a>

| Concept | Code |
|---------|------|
| Binary column `1` if any substring matches `https?://\S+` or `www.\S+`. | ```python<br>class UrlFlagTransformer(BaseEstimator, TransformerMixin):<br>    def __init__(self):<br>        self.regex = re.compile(r"https?://\S+|www\.\S+")<br>    def transform(self, X):<br>        flags = [1 if self.regex.search(txt) else 0 for txt in X]<br>        return sparse.csr_matrix(np.array(flags).reshape(-1,1))``` |

### 4️⃣ Domain whitelist <a name="4-domain-whitelist"></a>

If the URL belongs to a **trusted domain** (e.g. `amazon.com`, `paypal.com`) we set a **negative indicator**.

| Concept | Code |
|---------|------|
| Extract the first URL, normalise the domain, check membership in a hard‑coded whitelist. | ```python<br>class DomainWhitelistTransformer(BaseEstimator, TransformerMixin):<br>    def __init__(self, whitelist=None):<br>        self.whitelist = set(w.lower() for w in (whitelist or [...] ))<br>    def transform(self, X):<br>        pattern = re.compile(r"https?://([^/]+)")<br>        flags = []<br>        for txt in X:<br>            m = pattern.search(txt)<br>            if m:<br>                domain = m.group(1).lower().lstrip("www.")<br>                flags.append(1 if domain in self.whitelist else 0)<br>            else:<br>                flags.append(0)<br>        return sparse.csr_matrix(np.array(flags).reshape(-1,1))``` |

### 5️⃣ Keyword counter <a name="5-keyword-counter"></a>

| Concept | Code |
|---------|------|
| 120+ manually curated phishing‑related words/phrases (e.g. *“verify”, “account”, “urgent”, “$”*). | ```python<br>class KeywordFlagTransformer(BaseEstimator, TransformerMixin):<br>    def __init__(self, keywords=None):<br>        self.keywords = [...]  # (list from the file) <br>    def transform(self, X):<br>        rows = []<br>        for txt in X:<br>            txt_lc = txt.lower()<br>            rows.append([txt_lc.count(kw) for kw in self.keywords])<br>        return sparse.csr_matrix(rows)``` |

### 6️⃣ First‑person pronoun flag <a name="6-first-person-pronoun-flag"></a>

| Concept | Code |
|---------|------|
| Binary column `1` if any token belongs to `{i, me, my, we, our}`. | ```python<br>class FirstPersonPronounTransformer(BaseEstimator, TransformerMixin):<br>    def __init__(self):<br>        self.pronouns = {"i","me","my","we","our"}<br>    def transform(self, X):<br>        flags = [1 if set(txt.lower().split()) & self.pronouns else 0 for txt in X]<br>        return sparse.csr_matrix(np.array(flags).reshape(-1,1))``` |

All four custom transformers are **stateless** (no learning parameters), so they can be **pickled** together with the pipeline.

---

## Training pipeline (code) <a name="training-pipeline-code"></a>

```python
# train_spam.py
import re, pandas as pd
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from feature_transformers import (
    UrlFlagTransformer,
    KeywordFlagTransformer,
    DomainWhitelistTransformer,
    FirstPersonPronounTransformer,
)

# -----------------------------------------------------------------
# 0️⃣  Paths
# -----------------------------------------------------------------
CSV_PATH   = Path("data/spam.csv")
MODEL_PATH = Path("model/spam_detector.pkl")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------
# 1️⃣  Load & clean data
# -----------------------------------------------------------------
df = pd.read_csv(CSV_PATH, header=None, usecols=[0,1],
                 names=["label","text"], encoding="utf-8", engine="python")
df.dropna(subset=["label","text"], inplace=True)

label_map = {"ham":0, "legitimate":0, "spam":1, "phishing":1}
df["label"] = df["label"].map(label_map).astype(int)

def clean_text(t):
    return re.sub(r"\s+", " ", t.replace("\uFFFD","")).strip()
df["clean"] = df["text"].apply(clean_text)

# -----------------------------------------------------------------
# 2️⃣  Train / test split
# -----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label"], test_size=0.2,
    stratify=df["label"], random_state=42)

# -----------------------------------------------------------------
# 3️⃣  FeatureUnion + LogisticRegression
# -----------------------------------------------------------------
pipeline = Pipeline([
    ("features", FeatureUnion([
        ("tfidf_word",   TfidfVectorizer(ngram_range=(1,2),
                                         max_features=None,
                                         min_df=1,
                                         sublinear_tf=True,
                                         stop_words=None)),
        ("tfidf_char",   TfidfVectorizer(analyzer="char_wb",
                                         ngram_range=(3,5),
                                         min_df=1,
                                         sublinear_tf=True,
                                         max_features=5000)),
        ("url_flag",     UrlFlagTransformer()),
        ("keyword_flag", KeywordFlagTransformer()),
        ("whitelist",    DomainWhitelistTransformer()),
        ("first_person", FirstPersonPronounTransformer()),
    ])),
    ("clf", LogisticRegression(max_iter=3000,
                               class_weight="balanced",
                               solver="lbfgs"))
])

pipeline.fit(X_train, y_train)

# -----------------------------------------------------------------
# 4️⃣  Evaluation
# -----------------------------------------------------------------
from sklearn.metrics import classification_report, roc_auc_score
y_prob = pipeline.predict_proba(X_test)[:,1]
y_pred = (y_prob >= 0.5).astype(int)

print("\n=== Classification report ===")
print(classification_report(y_test, y_pred,
                            target_names=["ham","phishing"]))
print("ROC‑AUC :", roc_auc_score(y_test, y_prob))

# -----------------------------------------------------------------
# 5️⃣  Save the whole pipeline
# -----------------------------------------------------------------
joblib.dump(pipeline, MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")

