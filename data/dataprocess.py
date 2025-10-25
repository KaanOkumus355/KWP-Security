import pandas as pd

# 👉  Change the file name / path to where you keep the CSV
CSV_PATH = "dataset/spam.csv"

print("🔎 Loading CSV from:", CSV_PATH)

try:
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
except UnicodeDecodeError:
    # Try latin‑1 (Windows‑1252) – works for most non‑UTF8 CSVs
    df = pd.read_csv(CSV_PATH, encoding="latin-1")

df = df.rename(columns={
    "message_text": "text",      # <-- example
    "spam": "label"
})

# drop columns in DataFrame

df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4' ], axis=1, inplace=True)
# A quick sanity check
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print(df.head())
