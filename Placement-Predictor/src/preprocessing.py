import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add stable derived features used by both training and inference."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Work on a copy so the original data is not changed.
        df = pd.DataFrame(X).copy()

        # Create combined scores from existing student details.
        df["Skill_Score"] = (
            df["Coding_Skills"]
            + df["Communication_Skills"]
            + df["Soft_Skills_Rating"]
        )
        df["Experience_Score"] = df["Internships"] + df["Projects"]
        df["Academic_Strength"] = df["CGPA"] + df["Certifications"]

        return df
