# Real Estate ML Predictions
This repository contains code for predicting real estate property prices and property types using ensemble machine learning models. The project applies advanced preprocessing, feature engineering, and model ensembling to improve prediction accuracy.
- UNSW Final Assignment for Data Services Engineering Course, T1 2025 
- Achieved 100% assignment mark

## Project Overview

The project addresses two main tasks:  

1. **Regression Task** – Predict the sale price of properties based on features such as location, distances from CBD, public transport times, demographic breakdown, and property characteristics.  
2. **Classification Task** – Predict the property type (e.g., house, apartment) using the same set of features.  

The workflow leverages **XGBoost** and **LightGBM** models with ensembling for more robust predictions.

## Features

- Handles missing data via Random Forest imputation  
- Generates cyclical features for month and day to better capture temporal patterns  
- Applies transformations to skewed continuous features  
- Encodes categorical features using one-hot encoding  
- Ensemble prediction combines multiple models to improve accuracy  

## Technologies

- Python 3
- Pandas & NumPy
- Scikit-learn
- XGBoost
- LightGBM

## Usage

1. Clone the repository:  
```bash
git clone https://github.com/RyanFarahani1/real-estate-ml-predictions.git
cd real-estate-ml-predictions
```

2. Place train.csv and test.csv in the project folder.

Run the script:
```bash
python3 main.py train.csv test.csv
```

3. Outputs:

main.classification.csv – Predicted property types

main.regression.csv – Predicted property prices



## Evaluation

- Classification: F1-weighted score

- Regression: Mean Absolute Error (MAE)

- Ensemble predictions improve performance over individual models.

## Notes

- Rare classes filtered for classification

- Missing values in numeric columns are imputed

- Skewed features transformed when necessary

- Cyclical transformations capture seasonal patterns
