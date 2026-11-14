import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('dataset/Crop_recommendation.csv')
print("dataset loaded successfully")

"""# Display first five records

print("\nFirst Five Records:")
print(df.head())


# Display number of rows and columns

print("\nDataset Shape:")
print(df.shape)


# Display dataset information

print("\nDataset Information:")
df.info()

# Check missing values

print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate records

print("\nDuplicate Records:")
print(df.duplicated().sum())


# Display all available crops

print("\nAvailable Crops:")
print(df["label"].unique())

print("\nTotal Crop Categories:")
print(df["label"].nunique())


# Descriptive Statistics

print("\nDescriptive Statistics:")
print(df.describe())
"""

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import joblib


# Separate input features and target column

X = df.drop("label", axis=1)

y = df["label"]

print("\nInput Features:")
print(X.head())

print("\nTarget Output:")
print(y.head())


# Split dataset into training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# Create Decision Tree model

dt_model = DecisionTreeClassifier(random_state=42)

# Train the model

dt_model.fit(X_train, y_train)

print("\nDecision Tree Model Trained Successfully!")


# Create Random Forest model

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model

rf_model.fit(X_train, y_train)

print("\nRandom Forest Model Trained Successfully!")


# Predict using Decision Tree

dt_predictions = dt_model.predict(X_test)

# Predict using Random Forest

rf_predictions = rf_model.predict(X_test)

# Calculate accuracy

dt_accuracy = accuracy_score(y_test, dt_predictions)

rf_accuracy = accuracy_score(y_test, rf_predictions)

print("\nDecision Tree Accuracy:", dt_accuracy * 100, "%")

print("Random Forest Accuracy:", rf_accuracy * 100, "%")


# Compare both models

print("\nModel Comparison")

print("Decision Tree:", round(dt_accuracy * 100, 2), "%")

print("Random Forest:", round(rf_accuracy * 100, 2), "%")

if rf_accuracy > dt_accuracy:
    best_model = rf_model
    best_model_name = "Random Forest"

else:
    best_model = dt_model
    best_model_name = "Decision Tree"

print("\nBest Model:", best_model_name)

# Decision Tree evaluation

print("\nDecision Tree Classification Report")

print(classification_report(y_test, dt_predictions))


# Random Forest evaluation

print("\nRandom Forest Classification Report")

print(classification_report(y_test, rf_predictions))

# Confusion Matrix for best model

cm = confusion_matrix(y_test, best_model.predict(X_test))

plt.figure(figsize=(12, 9))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens"
)

plt.title("CropWise - Confusion Matrix")
plt.xlabel("Predicted Crop")
plt.ylabel("Actual Crop")

plt.tight_layout()
plt.show()


# Save the best model

joblib.dump(best_model, "model/crop_model.pkl")

print("\nBest Model Saved Successfully!")
