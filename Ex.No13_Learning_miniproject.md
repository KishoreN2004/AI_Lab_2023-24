# Ex.No: 10 Learning – Use Supervised Learning  
### DATE:  25/10/2025                                                                          
### REGISTER NUMBER : 212223065001
# Lab Experiment: Cancer Prediction Using Logistic Regression

## Aim
To build a machine learning model to predict cancer diagnosis based on patient features using Logistic Regression and demonstrate predictions using Gradio interface.

## Objective
- Load and preprocess cancer dataset.
- Train a Logistic Regression model.
- Use Gradio to take user input and predict cancer status.
- Demonstrate predictions on multiple sample inputs.

## Introduction
Cancer diagnosis is a critical application of AI in healthcare. Logistic Regression can be used to classify tumors as benign or malignant based on features like radius, texture, smoothness, etc. This experiment demonstrates a working prediction model with an interactive interface.

## Algorithm Used
Cancer Prediction Using Logistic Regression
**Input:** Patient tumor features (numerical values)  
**Output:** Cancer status – Malignant or Benign  

1. **Load Dataset:** Use `sklearn.datasets` breast cancer dataset. Separate features (`X`) and target labels (`y`).  
2. **Preprocess Data:** Split into training and testing sets. Standardize features using `StandardScaler`.  
3. **Train Model:** Initialize and train `LogisticRegression` on scaled training data.  
4. **Prediction Function:**  
   - Take input features from user  
   - Scale using the same `StandardScaler`  
   - Predict class using trained model  
   - Convert numeric output to labels: `0 → Malignant`, `1 → Benign`  
5. **Gradio Interface:** Create input fields for features, link the prediction function, display output as text.  
6. **Test Samples:** Provide predefined sample inputs to automatically show predictions. 

## Code

```python
# Import packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import gradio as gr

# Load dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Sample inputs (automatically included)
samples = [
    [17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189],
    [20.57, 17.77, 132.9, 1326, 0.08474, 0.07864, 0.0869, 0.07017, 0.1812, 0.05667, 0.5435, 0.7339, 3.398, 74.08, 0.005225, 0.01308, 0.0186, 0.0134, 0.01389, 0.003532, 24.99, 23.41, 158.8, 1956, 0.1238, 0.1866, 0.2416, 0.186, 0.275, 0.08902],
    [19.69, 21.25, 130, 1203, 0.1096, 0.1599, 0.1974, 0.1279, 0.2069, 0.05999, 0.7456, 0.7869, 4.5, 94.03, 0.00615, 0.03681, 0.04427, 0.01885, 0.02499, 0.009043, 23.57, 25.53, 152.5, 1575, 0.1444, 0.4245, 0.4504, 0.243, 0.3613, 0.08758]
]

# Function for Gradio prediction
def cancer_prediction(*features):
    x = np.array(features).reshape(1, -1)
    prediction = model.predict(scaler.transform(x))
    return "Malignant" if prediction[0]==0 else "Benign"

# Launch Gradio
inputs = [gr.Number(label=feature) for feature in feature_names]
app = gr.Interface(fn=cancer_prediction, inputs=inputs, outputs="text", description="Breast Cancer Prediction")
app.launch(share=True)

# Optional: Automatic predictions for predefined samples
print("Sample Predictions:")
for i, sample in enumerate(samples):
    pred = model.predict(scaler.transform([sample]))
    label = "Malignant" if pred[0]==0 else "Benign"
    print(f"Sample {i+1}: {label}")
```

## Output

<img width="1571" height="595" alt="image" src="https://github.com/user-attachments/assets/a757fb75-0d9a-425b-a133-e599721397b3" />
<img width="1590" height="384" alt="image" src="https://github.com/user-attachments/assets/b1edbce1-4163-4c81-b42a-d80107ccb621" />

## Observation

Logistic Regression successfully classified most samples correctly.

Standardizing features improved model performance.

Gradio interface allows easy testing with custom inputs.

## Result
This experiment demonstrates a working AI model for cancer prediction. Using Logistic Regression and Gradio, predictions can be made interactively or on predefined samples.
