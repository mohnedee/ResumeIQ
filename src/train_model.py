import pandas as pd
import joblib

from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Resume.csv"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

REPORTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# 1. Load and prepare the dataset
df = pd.read_csv(DATA_PATH)
df = df[["Resume_str", "Category"]].dropna()
df = df.drop_duplicates(subset=["Resume_str"])

X = df["Resume_str"]
y = df["Category"]

# 2. Split data into training, validation, and final test sets
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval,
    y_trainval,
    test_size=0.25,
    random_state=42,
    stratify=y_trainval
)

print(f"Total resumes: {len(df)}")
print(f"Training resumes: {len(X_train)}")
print(f"Validation resumes: {len(X_val)}")
print(f"Final test resumes: {len(X_test)}")

# 3. Define three candidate models
classifiers = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),
    "Linear SVM": LinearSVC(
        class_weight="balanced",
        random_state=42
    ),
    "Multinomial Naive Bayes": MultinomialNB()
}

def make_pipeline(classifier):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=20000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ("classifier", classifier)
    ])

# 4. Compare models using validation data only
results = []
best_score = -1
best_name = ""
best_validation_model = None

for name, classifier in classifiers.items():
    print(f"\nTraining: {name}")

    model = make_pipeline(classifier)
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(y_val, predictions, average="macro")
    weighted_f1 = f1_score(y_val, predictions, average="weighted")

    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Macro F1: {macro_f1:.4f}")
    print(f"Validation Weighted F1: {weighted_f1:.4f}")

    results.append({
        "Model": name,
        "Validation Accuracy": accuracy,
        "Validation Macro F1": macro_f1,
        "Validation Weighted F1": weighted_f1
    })

    if macro_f1 > best_score:
        best_score = macro_f1
        best_name = name
        best_validation_model = model

comparison = pd.DataFrame(results)
comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

print("\n=== VALIDATION MODEL COMPARISON ===")
print(comparison.to_string(index=False))
print(f"\nSelected model: {best_name}")
print(f"Validation Macro F1: {best_score:.4f}")

# 5. Retrain the selected model on training + validation data
final_model = clone(best_validation_model)
final_model.fit(X_trainval, y_trainval)

# 6. Evaluate once on the held-out final test set
test_predictions = final_model.predict(X_test)

test_metrics = {
    "Selected Model": best_name,
    "Test Accuracy": accuracy_score(y_test, test_predictions),
    "Test Macro F1": f1_score(
        y_test, test_predictions, average="macro", zero_division=0
    ),
    "Test Weighted F1": f1_score(
        y_test, test_predictions, average="weighted", zero_division=0
    ),
    "Training Samples": len(X_trainval),
    "Test Samples": len(X_test)
}

pd.DataFrame([test_metrics]).to_csv(
    REPORTS_DIR / "final_test_metrics.csv", index=False
)

test_report = classification_report(
    y_test, test_predictions, output_dict=True, zero_division=0
)
pd.DataFrame(test_report).transpose().to_csv(
    REPORTS_DIR / "classification_report.csv",
    index=True,
    index_label="Category"
)

joblib.dump(final_model, MODELS_DIR / "best_model.joblib")

print("\n=== FINAL TEST RESULTS ===")
for metric, value in test_metrics.items():
    if isinstance(value, float):
        print(f"{metric}: {value:.4f}")
    else:
        print(f"{metric}: {value}")

print(f"\nSaved validation comparison: {REPORTS_DIR / 'model_comparison.csv'}")
print(f"Saved final test metrics: {REPORTS_DIR / 'final_test_metrics.csv'}")
print(f"Saved classification report: {REPORTS_DIR / 'classification_report.csv'}")
print(f"Saved final model: {MODELS_DIR / 'best_model.joblib'}")
