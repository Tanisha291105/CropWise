<<<<<<< HEAD
# CropWise

CropWise is a Streamlit application that recommends a crop from soil nutrient and local weather readings. Enter your farm values to see the model's best match and a probability breakdown for all supported crops.

## Features

- Accepts nitrogen (N), phosphorus (P), potassium (K), soil pH, temperature, humidity, and rainfall.
- Recommends one crop without displaying its confidence score in the recommendation.
- Shows the model probabilities for all 22 supported crops in a ranked horizontal bar chart.
- Includes sample inputs and a session-based prediction history.
- Warns when an input is outside the range observed in the training dataset.
- Uses a simple light theme configured in `.streamlit/config.toml`.

CropWise does not fetch live weather data. Enter the weather readings you want to use for the recommendation.

## Supported crops

The model is trained to classify these 22 dataset labels:

`apple`, `banana`, `blackgram`, `chickpea`, `coconut`, `coffee`, `cotton`, `grapes`, `jute`, `kidneybeans`, `lentil`, `maize`, `mango`, `mothbeans`, `mungbean`, `muskmelon`, `orange`, `papaya`, `pigeonpeas`, `pomegranate`, `rice`, and `watermelon`.

## Requirements

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- streamlit
- joblib
Run commands from the CropWise project root. The app loads its model and dataset using paths relative to that directory.

## Setup on Windows

Open PowerShell in the project folder and create a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents environment activation, call the environment's executables directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the application

From the project root, start the Streamlit app:

```powershell
streamlit run app.py
```

If you did not activate the environment, use:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

In the **Crop Recommendation** page:

1. Enter soil test values for N, P, K, and pH.
2. Enter temperature, humidity, and rainfall values.
3. Select **Get my crop recommendation**.
4. Review the recommended crop. Expand **Compare all crop matches** to view the ranked probability chart.

Use **Try Sample Values** to populate an example set of values. **Prediction History** stores predictions for the current Streamlit session; it is not a permanent database.

## Test a prediction and evaluate the model

Run the interactive test script from the project root:

```powershell
python test_model.py
```

The script prompts for the same seven soil and weather inputs. Enter a number at each prompt, or press Enter to use the displayed sample value. It prints a crop prediction and confidence, then reports accuracy and a classification report for the test split.

Without environment activation:

```powershell
.\.venv\Scripts\python.exe test_model.py
```

## Retrain the model

The training script compares a Decision Tree with a 100-estimator Random Forest using a stratified 80/20 train/test split. It displays the model metrics and a confusion matrix, then saves the better-performing model to `model/crop_model.pkl`.

```powershell
python main.py
```

Retraining overwrites the existing model file. Run this from the project root so the relative dataset and model paths resolve correctly.

## Dataset and model

`dataset/Crop_recommendation.csv` contains 2,200 rows with seven input features (`N`, `P`, `K`, `temperature`, `humidity`, `ph`, and `rainfall`) and a target `label` identifying one of the 22 crops. The trained model is stored at `model/crop_model.pkl`.

Model confidence is the predicted class probability, not a guarantee of crop success or yield. Recommendations are limited to the crops and conditions represented in the dataset. Use reliable soil test results and local agricultural guidance when making planting decisions.

## Project structure

```text
CropWise/
├── .streamlit/
│   └── config.toml                 # Streamlit theme and minimal toolbar
├── dataset/
│   └── Crop_recommendation.csv     # Training and evaluation data
├── model/
│   └── crop_model.pkl              # Trained classifier used by the app
├── app.py                          # Streamlit user interface
├── main.py                         # Model training and evaluation
├── test_model.py                   # Interactive prediction and test evaluation
└── requirements.txt                # Python dependencies
```
=======
# CropWise
>>>>>>> ed67f4f60590b9e8737365e7b814d3405c5a4f89
