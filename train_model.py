import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("=" * 50)
print("Loading Stroke Dataset...")
print("=" * 50)

# -------------------------------
# Load Dataset
# -------------------------------

data = pd.read_csv("dataset/healthcare-dataset-stroke-data.csv")

print("Dataset Loaded Successfully!")
print(data.head())

# -------------------------------
# Remove ID Column
# -------------------------------

if "id" in data.columns:
    data.drop("id", axis=1, inplace=True)

# -------------------------------
# Handle Missing Values
# -------------------------------

data["bmi"] = data["bmi"].fillna(data["bmi"].median())

# -------------------------------
# Separate Features and Target
# -------------------------------

X = data.drop("stroke", axis=1)
y = data["stroke"]

# -------------------------------
# Convert Categorical Data
# -------------------------------

X = pd.get_dummies(X)

# Save feature names
joblib.dump(X.columns.tolist(), "model_columns.pkl")

# -------------------------------
# Split Dataset
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# -------------------------------
# Train Model
# -------------------------------

print("\nTraining Random Forest Model...\n")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# Accuracy
# -------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# -------------------------------
# Save Model
# -------------------------------

joblib.dump(model, "stroke_model.pkl")

print("\nModel saved successfully!")
print("stroke_model.pkl created.")
print("model_columns.pkl created.")

print("=" * 50)
print("Training Completed Successfully!")
print("=" * 50)