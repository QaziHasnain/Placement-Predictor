from pathlib import Path
import sys

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
MODEL_PATH = BASE_DIR / "models" / "placement_model.pkl"
COLUMNS_PATH = BASE_DIR / "models" / "model_columns.pkl"

# Make sure local src modules can be imported when this file runs directly.
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


def cap_probability(probability):
    # Keep very high probabilities from looking overconfident.
    if probability > 0.95:
        return 0.92
    return probability


def predict_placement(student_profile):
    # Load the trained model and the column order used during training.
    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(COLUMNS_PATH)

    # Convert the input dictionary into a one-row dataframe for prediction.
    input_df = pd.DataFrame([student_profile], columns=model_columns)

    # Predict placement probability and convert it into a readable result.
    probability = cap_probability(float(model.predict_proba(input_df)[0][1]))
    result = "PLACED" if probability >= 0.5 else "NOT PLACED"

    return {
        "result": result,
        "placement_probability": round(probability * 100, 1),
    }


if __name__ == "__main__":
    # Example student data for testing this file directly.
    sample = {
        "Age": 21,
        "Gender": "Male",
        "Degree": "B.Tech",
        "Branch": "CSE",
        "CGPA": 8.5,
        "Internships": 1,
        "Projects": 2,
        "Coding_Skills": 7,
        "Communication_Skills": 7,
        "Aptitude_Test_Score": 70,
        "Soft_Skills_Rating": 7,
        "Certifications": 1,
        "Backlogs": 0,
    }
    print(predict_placement(sample))
