
import pandas as pd
import joblib
import math


def read_float(prompt, default):
    while True:
        value = input(f"{prompt} [{default}]: ").strip()

        if not value:
            return default

        try:
            number = float(value)
        except ValueError:
            print("Please enter a numeric value.")
            continue

        if not math.isfinite(number):
            print("Please enter a finite number.")
            continue

        return number

# Load trained model

model = joblib.load("model/crop_model.pkl")

print("CropWise Model Loaded Successfully!")
print("\nEnter your soil and weather values.")
print("Press Enter to use the sample value shown in brackets.\n")

# Collect soil and weather values from the user

nitrogen = read_float("Nitrogen (N)", 90)
phosphorus = read_float("Phosphorus (P)", 42)
potassium = read_float("Potassium (K)", 43)
temperature = read_float("Temperature (°C)", 20.8)
humidity = read_float("Humidity (%)", 82.0)
ph = read_float("Soil pH", 6.5)
rainfall = read_float("Rainfall (mm)", 202.9)

# Create input data

input_data = pd.DataFrame(
    [[nitrogen, phosphorus, potassium,
      temperature, humidity, ph, rainfall]],
    columns=[
        "N", "P", "K",
        "temperature", "humidity",
        "ph", "rainfall"
    ]
)

# Predict crop

prediction = model.predict(input_data)

print("\nMost Suitable Crop:", prediction[0])

# Calculate prediction confidence

probabilities = model.predict_proba(input_data)

confidence = probabilities.max() * 100

print("Prediction Confidence:", round(confidence, 2), "%")


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Load original dataset

df = pd.read_csv("dataset/Crop_recommendation.csv")

# Separate input and output

X = df.drop("label", axis=1)
y = df["label"]

# Recreate the same test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Predict test data

test_predictions = model.predict(X_test)

# Calculate accuracy

accuracy = accuracy_score(y_test, test_predictions)

print("\nTest Dataset Accuracy:", round(accuracy * 100, 2), "%")

# Display classification report

print("\nClassification Report:")
print(classification_report(y_test, test_predictions))
