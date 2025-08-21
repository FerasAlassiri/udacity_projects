# developer_salary_analysis.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import re

# -------------------------
# STEP 1: Load Data
# -------------------------
print("📂 Loading dataset...")
df = pd.read_csv("survey_results_public.csv")

print("\n✅ Data Loaded Successfully")
print("Shape of dataset:", df.shape)
print(df.head())

# -------------------------
# STEP 2: Explore Data
# -------------------------
print("\n📊 Dataset Info:")
print(df.info())

print("\n📊 Summary Statistics:")
print(df.describe(include="all"))


# -------------------------
# STEP 3: Data Cleaning
# -------------------------

# Drop rows without salary
df = df.dropna(subset=["ConvertedCompYearly"])
df = df[df["ConvertedCompYearly"] > 0]

# Target variable: HighIncome
df["HighIncome"] = np.where(df["ConvertedCompYearly"] > 100000, 1, 0)

# Parse YearsCode
def parse_years_code(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if s.lower().startswith("less than"): return 0.5
    if s.lower().startswith("more than"): return 50
    try:
        return float(s)
    except:
        return np.nan

df["YearsCodeClean"] = df["YearsCode"].apply(parse_years_code)

# Parse Age
def parse_age(x):
    if pd.isna(x): return np.nan
    s = str(x)
    if "Under" in s: return 17
    if "65" in s: return 70
    if "-" in s:
        parts = s.split("-")
        return (int(parts[0]) + int(parts[1].split()[0])) / 2
    try:
        return int(re.search(r"\d+", s).group())
    except:
        return np.nan

df["AgeClean"] = df["Age"].apply(parse_age)

# Features & target
X = df[["YearsCodeClean", "AgeClean"]]
y = df["HighIncome"]

# Handle NaNs
X = X.fillna(X.median())

# -------------------------
# STEP 4: Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# STEP 5: Logistic Regression
# -------------------------
print("\n🤖 Training Logistic Regression Model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n✅ Model Trained Successfully")

# -------------------------
# STEP 6: Evaluation
# -------------------------
print("🔎 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))