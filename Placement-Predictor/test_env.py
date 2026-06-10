import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Create a dummy dataset to simulate placement data
df = pd.read_csv('../data/test.csv')

df = pd.DataFrame({
    'cgpa': [8.5, 7.2, 9.0, 6.8, 8.0],
    'internships': [2, 0, 3, 1, 4],
    'placed': [1, 0, 1, 0, 1]})
print("--- Dataset Sample ---")
print(df.head())

# 2. Split data into Features (X) and Target (y)
X = df[['cgpa', 'internships']]
y = df['placed']

# 3. Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train the Model
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)

# 5. Make a manual prediction
# Let's predict for a student with CGPA 8.2 and 1 internship
sample_student = pd.DataFrame([[8.0, 3]], columns=['cgpa', 'internships'])
prediction = model.predict(sample_student)

result = "Placed" if prediction[0] == 1 else "Not Placed"
print(f"\n--- Test Prediction ---\nResult: {result}")
