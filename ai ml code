import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# ----------------------------------------------------
# 1. Convert Excel → CSV (RUN ONCE)
# ----------------------------------------------------
# Convert Excel → CSV
excel_path = "DATASET.xlsx"
csv_path = "DATASET.csv"

data_excel = pd.read_excel(excel_path)
data_excel.to_csv(csv_path, index=False)

print("Excel converted to CSV successfully!")

# ----------------------------------------------------
# 2. Load CSV Dataset
# ----------------------------------------------------
data = pd.read_csv(csv_path)

print("Dataset Loaded Successfully!")
print(data.head())
print("Columns:", data.columns.tolist())

# ----------------------------------------------------
# 3. Detect Label Column
# ----------------------------------------------------
label_candidates = [c for c in data.columns if c.lower() == "label"]

if label_candidates:
    label_col = label_candidates[0]
else:
    bin_cols = [c for c in data.columns if set(data[c].dropna().unique()).issubset({0, 1})]
    if len(bin_cols) == 1:
        label_col = bin_cols[0]
    else:
        raise KeyError(f"Label column not found. Columns: {data.columns.tolist()}")

print(f"Using label column: {label_col}")

# ----------------------------------------------------
# 4. Split Features & Target
# ----------------------------------------------------
X = data.drop(label_col, axis=1)
y = data[label_col]

# Identify column types
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

# ----------------------------------------------------
# 5. Preprocessing
# ----------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ]
)

# ----------------------------------------------------
# 6. Model Pipeline
# ----------------------------------------------------
model = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', LogisticRegression(max_iter=200, class_weight='balanced'))
])

# ----------------------------------------------------
# 7. Train-Test Split
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ----------------------------------------------------
# 8. Train Model
# ----------------------------------------------------
model.fit(X_train, y_train)

# ----------------------------------------------------
# 9. Predictions
# ----------------------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ----------------------------------------------------
# 10. Evaluation
# ----------------------------------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nROC-AUC Score:", roc_auc_score(y_test, y_prob))

# ----------------------------------------------------
# 11. Sample Prediction
# ----------------------------------------------------
sample = X.iloc[[0]]
prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0][1]

print("\nSample Prediction:", prediction)
print("Hazardous Probability:", round(probability * 100, 2), "%")
