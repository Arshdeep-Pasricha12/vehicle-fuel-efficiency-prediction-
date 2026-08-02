"""
predict.py — Standalone prediction script for the Vehicle Fuel Efficiency model

Usage:
    python src/predict.py
"""

import pandas as pd
import numpy as np
import joblib

MODEL_PATH = 'models/final_model.pkl'

# Feature order must match training
FEATURE_COLS = [
    'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration',
    'model_year', 'power_to_weight', 'displacement_per_cyl',
    'log_weight', 'log_displacement', 'log_horsepower',
    'weight_class_enc', 'era_enc', 'origin_2', 'origin_3',
]


def preprocess_input(sample: dict) -> pd.DataFrame:
    """
    Preprocess a raw vehicle record into model-ready features.

    Parameters
    ----------
    sample : dict with keys: cylinders, displacement, horsepower,
             weight, acceleration, model_year, origin (1/2/3)

    Returns
    -------
    pd.DataFrame — single-row feature matrix
    """
    s = sample.copy()

    # Engineered features
    s['power_to_weight']      = s['horsepower'] / s['weight']
    s['displacement_per_cyl'] = s['displacement'] / s['cylinders']
    s['log_weight']           = np.log(s['weight'])
    s['log_displacement']     = np.log(s['displacement'])
    s['log_horsepower']       = np.log(s['horsepower'])

    # Weight class (ordinal: Light=0, Mid=1, Heavy=2, Very Heavy=3)
    w = s['weight']
    if w < 2500:    s['weight_class_enc'] = 0
    elif w < 3500:  s['weight_class_enc'] = 1
    elif w < 4500:  s['weight_class_enc'] = 2
    else:           s['weight_class_enc'] = 3

    # Era encoding (0=Pre-Crisis, 1=Crisis, 2=Post-Crisis)
    yr = s['model_year']
    if yr <= 73:    s['era_enc'] = 0
    elif yr <= 78:  s['era_enc'] = 1
    else:           s['era_enc'] = 2

    # Origin one-hot (drop first = origin 1/USA as reference)
    s['origin_2'] = 1 if s['origin'] == 2 else 0
    s['origin_3'] = 1 if s['origin'] == 3 else 0

    df = pd.DataFrame([s])[FEATURE_COLS]
    return df


def predict_mpg(sample: dict) -> float:
    """Predict MPG for a single vehicle record."""
    model = joblib.load(MODEL_PATH)
    X = preprocess_input(sample)
    prediction = model.predict(X)[0]
    return round(prediction, 2)


if __name__ == '__main__':
    # Example vehicle
    example_vehicle = {
        'cylinders':    4,
        'displacement': 110,
        'horsepower':   88,
        'weight':       2400,
        'acceleration': 17.5,
        'model_year':   80,   # 1980
        'origin':       3,    # Japan
    }

    print('=== Vehicle Fuel Efficiency Predictor ===')
    print('\nInput Vehicle:')
    for k, v in example_vehicle.items():
        print(f'  {k:<16}: {v}')

    predicted_mpg = predict_mpg(example_vehicle)
    print(f'\n🚗 Predicted MPG: {predicted_mpg}')

    # Efficiency band
    if predicted_mpg < 20:
        band = '🔴 Low Efficiency'
    elif predicted_mpg < 28:
        band = '🟡 Mid Efficiency'
    else:
        band = '🟢 High Efficiency'

    print(f'📊 Efficiency Band: {band}')
