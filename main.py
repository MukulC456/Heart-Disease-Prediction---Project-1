import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Project1-dataset.csv")

# Define features and target variable
X = df.drop(columns=["target"])  # Features (all except target)
y = df["target"]  # Target labels

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model & scaler
with open("heart_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Feature importance plot
plt.figure(figsize=(10, 5))
sns.barplot(x=X.columns, y=model.feature_importances_)
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.show()


# --------------------
# Prediction Function
# --------------------
selected_features = "age,sex,chest pain type,resting bp s,cholesterol,fasting blood sugar,resting ecg,max heart rate,exercise angina,oldpeak,ST slope,target"
def predict_heart_disease(input_data):
    

    # Load model and scaler
    with open("heart_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Validate input length
    if len(input_data) != len(selected_features):
        raise ValueError(f"Expected {len(selected_features)} features, but got {len(input_data)}")

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data], columns=selected_features)

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)

    return "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"

# --------------------
# Sample Test Prediction
# --------------------
sample_input = [40,1,2,140,289,0,0,172,0,0,1,0]  # Example input based on UI features
print("Sample Prediction:", predict_heart_disease(sample_input))
