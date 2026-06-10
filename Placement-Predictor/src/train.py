from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import FeatureEngineer


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "train.csv"
MODEL_PATH = BASE_DIR / "models" / "placement_model.pkl"
COLUMNS_PATH = BASE_DIR / "models" / "model_columns.pkl"
REPORT_PATH = BASE_DIR / "models" / "training_report.txt"
TARGET_COLUMN = "Placement_Status"

DIRECT_LEAKAGE_COLUMNS = {
    "Salary",
    "Package",
    "Placement_Offer",
    "Offer_Letter",
    "Interview_Result",
    "Company_Placed",
    "PlacementStatus",
}


def load_training_data():
    # Read the training data and clean column names.
    df = pd.read_csv('../data/train.csv')
    df.columns = df.columns.str.strip()

    # Remove duplicate rows so repeated records do not affect training.
    original_rows = len(df)
    duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # Remove student id because it should not be used for prediction.
    dropped_columns = []
    if "Student_ID" in df.columns:
        dropped_columns.append("Student_ID")
        df = df.drop(columns=["Student_ID"])

    # Remove columns that directly reveal placement outcome.
    leakage_columns = sorted((DIRECT_LEAKAGE_COLUMNS & set(df.columns)) - {TARGET_COLUMN})
    if leakage_columns:
        dropped_columns.extend(leakage_columns)
        df = df.drop(columns=leakage_columns)

    # Convert placement labels into numbers for the model.
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing required target column: {TARGET_COLUMN}")

    y = df[TARGET_COLUMN].map({"Placed": 1, "Not Placed": 0})
    if y.isna().any():
        bad_values = sorted(df.loc[y.isna(), TARGET_COLUMN].dropna().unique().tolist())
        raise ValueError(f"Unexpected target values in {TARGET_COLUMN}: {bad_values}")

    X = df.drop(columns=[TARGET_COLUMN])

    diagnostics = {
        "original_rows": original_rows,
        "rows_after_dedup": len(df),
        "duplicate_rows_removed": duplicate_rows,
        "dropped_columns": dropped_columns,
        "class_counts": y.value_counts().sort_index().to_dict(),
    }
    return X, y, diagnostics


def build_pipeline():
    # Columns are split because numbers and categories need different handling.
    categorical_features = ["Gender", "Degree", "Branch"]
    numeric_features = [
        "Age",
        "CGPA",
        "Internships",
        "Projects",
        "Coding_Skills",
        "Communication_Skills",
        "Aptitude_Test_Score",
        "Soft_Skills_Rating",
        "Certifications",
        "Backlogs",
        "Skill_Score",
        "Experience_Score",
        "Academic_Strength",
    ]

    # Scale numeric values and one-hot encode category values.
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # Logistic regression is used as a simple placement classifier.
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def top_model_weights(model):
    # Show the strongest features learned by the model.
    classifier = model.named_steps["model"]
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    coefficients = classifier.coef_[0]
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefficients,
                "absolute_weight": abs(coefficients),
            }
        )
        .sort_values("absolute_weight", ascending=False)
        .head(15)
    )


X, y, diagnostics = load_training_data()

# Split the data so the model is tested on unseen records.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Build, validate, and train the model.
model = build_pipeline()
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1", n_jobs=1)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate model performance on the test data.
train_accuracy = model.score(X_train, y_train)
test_accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Not Placed", "Placed"])
matrix = confusion_matrix(y_test, y_pred)
model_weights = top_model_weights(model)

gap = train_accuracy - test_accuracy
cv_scores_display = [round(float(score), 4) for score in cv_scores]
warning = ""
if test_accuracy > 0.95 and abs(gap) < 0.02:
    warning = (
        "\nWarning:\n"
        "The model still scores unusually high even with split-safe preprocessing "
        "and a smooth regularized estimator. This usually means the dataset target "
        "is synthetic, rule-generated, or too easy. Validate with real placement data "
        "before presenting this as real-world accuracy.\n"
    )

# Create a text report with data checks, scores, and model details.
summary = f"""Data diagnostics:
Original rows: {diagnostics["original_rows"]}
Rows after deduplication: {diagnostics["rows_after_dedup"]}
Duplicate rows removed: {diagnostics["duplicate_rows_removed"]}
Dropped leakage/id columns: {diagnostics["dropped_columns"] or "None"}
Class counts: {diagnostics["class_counts"]}

Model:
LogisticRegression(
    C=1.0,
    class_weight="balanced",
    max_iter=2000,
    random_state=42
)

Cross-validation:
5-fold train F1 scores: {cv_scores_display}
Mean CV F1: {cv_scores.mean():.4f}
Std CV F1: {cv_scores.std():.4f}

Train vs test:
Train accuracy: {train_accuracy:.4f}
Test accuracy: {test_accuracy:.4f}
Accuracy gap: {gap:.4f}

Test metrics:
Accuracy: {test_accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1-score: {f1:.4f}

Confusion matrix:
{matrix}

Classification report:
{report}

Top model weights:
{model_weights.to_string(index=False)}
{warning}"""

print(summary)

# Save the model, input columns, and training report for later use.
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(X.columns.tolist(), COLUMNS_PATH)
REPORT_PATH.write_text(summary, encoding="utf-8")

print(f"\nModel saved to {MODEL_PATH}")
print(f"Training report saved to {REPORT_PATH}")
