from pathlib import Path

import joblib
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Resume.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

df = pd.read_csv(DATA_PATH)
df = df[["Resume_str", "Category"]].dropna()
df = df.drop_duplicates(subset=["Resume_str"])

X = df["Resume_str"]
y = df["Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

base_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )),
    ("classifier", LinearSVC(
        class_weight="balanced",
        random_state=42,
    )),
])

calibrated_model = CalibratedClassifierCV(
    estimator=base_model,
    method="sigmoid",
    cv=3,
)

print("Training calibrated LinearSVC. This may take a while...")
calibrated_model.fit(X_train, y_train)

predictions = calibrated_model.predict(X_test)
probabilities = calibrated_model.predict_proba(X_test)

print("\n=== CALIBRATED MODEL TEST RESULTS ===")
print(f"Test accuracy: {accuracy_score(y_test, predictions):.4f}")
print(
    "Test macro F1: "
    f"{f1_score(y_test, predictions, average='macro', zero_division=0):.4f}"
)
print(
    "Test weighted F1: "
    f"{f1_score(y_test, predictions, average='weighted', zero_division=0):.4f}"
)

example = X_test.iloc[0]
predicted_index = probabilities[0].argmax()

print("\n=== EXAMPLE CALIBRATED PREDICTION ===")
print(f"Predicted category: {calibrated_model.classes_[predicted_index]}")
print(f"Estimated confidence: {probabilities[0][predicted_index] * 100:.2f}%")

output_path = MODELS_DIR / "best_model_calibrated.joblib"
joblib.dump(calibrated_model, output_path)

print(f"\nCalibrated model saved separately: {output_path}")
print("The currently active model has NOT been replaced.")
