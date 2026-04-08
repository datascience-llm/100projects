# -*- coding: utf-8 -*-
"""
Linear Regression Learning_1

This script demonstrates:
- Loading a dataset
- Preprocessing features
- Training a Linear Regression model
- Making predictions
- Evaluating model performance
"""

# Import required libraries
import numpy as np                  # For numerical computations
import pandas as pd                 # For data handling (not strictly needed here but good practice)
import matplotlib.pyplot as plt     # For plotting (not used in this script)

# Load Iris dataset from sklearn
from sklearn.datasets import load_iris

dp = load_iris()        # Load dataset as a dictionary-like object
x = dp.data             # Feature matrix (sepal length, sepal width, etc.)
y = dp.target           # Target labels (0, 1, 2)

# -----------------------------
# Data Preprocessing
# -----------------------------

# Feature scaling helps many ML models perform better
from sklearn.preprocessing import StandardScaler

Scaler = StandardScaler()           # Create scaler object
x_Scaled = Scaler.fit_transform(x) # Scale features to mean=0 and std=1

# -----------------------------
# Model Selection
# -----------------------------

from sklearn.linear_model import LinearRegression

# Initialize Linear Regression model
model = LinearRegression()

# -----------------------------
# Model Training
# -----------------------------

# Train model using original features and target
model.fit(x, y)

# -----------------------------
# Prediction
# -----------------------------

# Predict output values using trained model
prediction = model.predict(x)

# Print raw prediction values
print(prediction)

# -----------------------------
# Model Evaluation
# -----------------------------

from sklearn.metrics import accuracy_score

# Linear Regression produces continuous values,
# so we round them to nearest class label (0, 1, 2)
abc = np.round(prediction)

# Calculate accuracy by comparing actual labels and predicted labels
accuracy = accuracy_score(y, abc)

# Print model accuracy
print("Accuracy:", accuracy)
