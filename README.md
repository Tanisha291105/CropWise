# CropWise

**Live application:** [Open CropWise](https://cropwise-bdu62d8rysyc9kappb6fhmp.streamlit.app/)

CropWise is a crop recommendation application that helps farmers and agronomists estimate the best crop to plant using soil nutrients and weather conditions. The app collects values for nitrogen, phosphorus, potassium, soil pH, temperature, humidity, and rainfall, predicts the most suitable crop, and shows the confidence of all supported crop matches.

This project uses a trained machine learning model on agricultural field data and provides a simple, interactive interface for Fast recommendations.

## Overview

CropWise is designed for quick crop selection decisions based on farm input values. It does not fetch live weather data; instead, the user enters relevant readings manually. The application compares all supported crop classes and ranks them by predicted probability so that the recommendation is interpretable.

## Features

- Accepts seven input features:
  - Nitrogen (N)
  - Phosphorus (P)
  - Potassium (K)
  - Soil pH
  - Temperature (°C)
  - Humidity (%)
  - Rainfall (mm)
- Recommends the most likely crop based on a trained classifier.
- Displays ranked crop probabilities for all 22 supported crops in a horizontal bar chart.
- Shows a description and emoji for the recommended crop.
- Includes a sample values button that fills in a realistic example dataset.
- Warns when an input falls outside the minimum and maximum range seen in the training dataset.
- Stores prediction history during the current Streamlit session.
- Provides a separate Prediction History page with total predictions and a clear-history option.
- Uses a light green agronomy-themed UI configured in `.streamlit/config.toml`.

## Tech Stack

- Python
- Streamlit
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

## Supported Crops

The model is trained to classify these 22 crops:

apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, jute, kidneybeans, lentil, maize, mango, mothbeans, mungbean, muskmelon, orange, papaya, pigeonpeas, pomegranate, rice, watermelon

## Project Structure

```text
CropWise/
├── .streamlit/
│   └── config.toml                 # Light theme and minimal Streamlit toolbar
├── dataset/
│   └── Crop_recommendation.csv     # Agricultural dataset used for training and testing
├── model/
│   └── crop_model.pkl              # Saved trained classifier
├── app.py                          # Streamlit web app UI and prediction logic
├── main.py                         # Model training and evaluation script
├── test_model.py                   # CLI prediction and test evaluation script
├── requirements.txt                # Python dependencies
├── README.md                      # Project documentation
└── .venv/                         # Optional local virtual environment (not tracked in git)
```

## Requirements

Make sure Python is installed, then install the project dependencies from the project root:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Setup on Windows

Open PowerShell in the CropWise folder and create a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run commands from the project root so relative paths work correctly.

## Run the Application

Start the Streamlit application:

```powershell
streamlit run app.py
```

If the virtual environment is not activated, use:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

### Using the app

1. Open the CropWise app in the browser.
2. On the Crop Recommendation page, enter farm values for N, P, K, pH, temperature, humidity, and rainfall.
3. Click `Get my crop recommendation`.
4. Review the recommended crop and the probability ranking under `Compare all crop matches`.
5. Use `Try Sample Values` to load a standard example input.
6. View prior predictions in the `Prediction History` page.

Prediction history is saved only in the current Streamlit session and is not a permanent database.

## Test a Prediction and Evaluate the Model

Run the interactive prediction script:

```powershell
python test_model.py
```

The script prompts for crop inputs and accepts either a value or a blank entry to use the default sample value. It prints:

- The predicted crop
- Prediction confidence (%)
- Test dataset accuracy
- Classification report for the evaluated model

If the virtual environment is not activated:

```powershell
.\.venv\Scripts\python.exe test_model.py
```

## Retrain the Model

The training script compares a Decision Tree model and a 100-estimator Random Forest classifier using a stratified 80/20 train/test split. It evaluates both models, chooses the better-performing one, and saves it to `model/crop_model.pkl`.

Run it with:

```powershell
python main.py
```

This command overwrites the saved model file. Always run it from the project root so the dataset and model paths resolve correctly.

### Machine learning models

The training script compares two supervised classification algorithms:

- **Decision Tree** (`DecisionTreeClassifier`): a single tree that learns a sequence of feature-based decisions.
- **Random Forest** (`RandomForestClassifier`): an ensemble of 100 decision trees whose predictions are combined.

Both models are evaluated on the same stratified 80/20 train/test split (with `random_state=42`). The script saves whichever model achieves the higher test accuracy; if their accuracies tie, it saves the Decision Tree. The model currently stored in `model/crop_model.pkl` is a **Random Forest**. After retraining, the selected model may change based on the evaluation results.

## Dataset and Model Details

The dataset file is:

```text
dataset/Crop_recommendation.csv
```

It contains 2,200 rows with the following feature columns:

- `N`
- `P`
- `K`
- `temperature`
- `humidity`
- `ph`
- `rainfall`

The target column is:

- `label`

The script loads the data, splits it into training and testing sets, compares model performance, and saves the best model into:

```text
model/crop_model.pkl
```

### Model notes

- Confidence is based on predicted class probability.
- The recommendation is limited to the crops and conditions represented in the dataset.
- Predictions may be less reliable if you enter values outside the training data range.
- Use professional agronomy advice and actual soil tests when making planting decisions.

## App Behavior Details

The app includes several practical checks before and after prediction:

- It validates input column order against the model's expected feature list.
- It compares each input against the dataset's observed minimum and maximum range.
- It shows a warning when a value is outside the training range.
- It ranks all crop probability scores from highest to lowest.
- It stores a prediction history for the active session.

## UI Theme

The application uses a light green theme with a minimal toolbar configuration defined in:

```toml
.streamlit/config.toml
```

This file sets:

- light base theme
- green primary accent
- white secondary backgrounds
- minimal toolbar mode

## Example Inputs

The app also includes a sample dataset button with the following values:

- Nitrogen: 90.0
- Phosphorus: 42.0
- Potassium: 43.0
- Temperature: 20.8°C
- Humidity: 82.0%
- Soil pH: 6.5
- Rainfall: 202.9 mm

These values are useful for testing the recommendation flow quickly.

## Notes

- CropWise does not connect to a live weather API.
- Weather values must be manually entered by the user.
- The prediction history is session-based only and resets when the session ends.
- The model is intended as a decision-support tool rather than a final agronomic guarantee.

## License

This project is intended for educational and personal use. If you are deploying it for a production or commercial environment, verify the licensing and data-use constraints of the dataset and dependencies before use.

## Quick Start

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```
