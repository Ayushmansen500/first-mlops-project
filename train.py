# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Create IT career dataset
data = {
    "Experience": [1,2,3,4,5,6,2,3,7,8],
    "CurrentPackage": [4,6,8,10,12,15,5,9,18,22],
    "SkillsCount": [2,3,4,5,6,7,2,4,8,9],
    "Certifications": [0,1,1,2,3,4,0,1,5,6],
    "CodingLevel": [3,4,5,6,7,8,4,6,9,9],
    "Outcome": [1,1,1,0,0,0,1,1,0,0]
}

df = pd.DataFrame(data)

print("✅ Columns:", df.columns.tolist())

# Prepare data
X = df[["Experience", "CurrentPackage", "SkillsCount", "Certifications", "CodingLevel"]]
y = df["Outcome"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "it_package_model.pkl")
print("✅ Model saved as it_package_model.pkl")
