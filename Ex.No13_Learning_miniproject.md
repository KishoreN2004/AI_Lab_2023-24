# Ex.No: 13 Learning – Use Supervised Learning  
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

```
# Step 1: Install packages
!pip install gradio scikit-learn pandas numpy --quiet

# Step 2: Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import gradio as gr

# Step 3: Simulate Lung Cancer dataset
data_dict = {
    "Age": [45, 60, 30, 50, 65, 40, 70, 55, 35, 28],
    "Smoking": [1,1,0,1,1,0,1,0,0,0],
    "Alcohol": [1,0,0,1,1,0,1,0,0,0],
    "Obesity": [0,1,0,1,1,0,1,0,0,0],
    "Family_History": [1,1,0,1,1,0,1,0,0,0],
    "Cancer": [1,1,0,1,1,0,1,0,0,0]
}
data = pd.DataFrame(data_dict)

# Step 4: Split dataset
x = data.drop(['Cancer'], axis=1)
y = data['Cancer']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

# Step 5: Scale features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Step 6: Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(x_train_scaled, y_train)

# Step 6.1: Compute accuracy
y_pred = model.predict(x_test_scaled)
model_accuracy = accuracy_score(y_test, y_pred)

# Step 7: Prediction function
def lung_cancer_prediction(Age, Smoking, Alcohol, Obesity, Family_History):
    try:
        x_input = np.array([
            Age if Age is not None else 0,
            Smoking if Smoking is not None else 0,
            Alcohol if Alcohol is not None else 0,
            Obesity if Obesity is not None else 0,
            Family_History if Family_History is not None else 0
        ]).reshape(1, -1)
        x_input_scaled = scaler.transform(x_input)
        pred = model.predict(x_input_scaled)
        return "YES" if pred[0] == 1 else "NO"
    except:
        return "Invalid Input"

# Step 8: Gradio Interface for user input
with gr.Blocks() as demo:
    gr.Markdown(f"## Lung Cancer Prediction\n**Model Accuracy:** {model_accuracy*100:.2f}%")
    
    with gr.Row():
        age_input = gr.Number(label="Age")
        smoking_input = gr.Number(label="Smoking (0 or 1)")
        alcohol_input = gr.Number(label="Alcohol (0 or 1)")
        obesity_input = gr.Number(label="Obesity (0 or 1)")
        family_input = gr.Number(label="Family History (0 or 1)")
    
    prediction_output = gr.Textbox(label="Prediction")
    
    btn = gr.Button("Predict")
    btn.click(
        fn=lung_cancer_prediction, 
        inputs=[age_input, smoking_input, alcohol_input, obesity_input, family_input],
        outputs=prediction_output
    )

demo.launch(share=True)
```

## Output

<img width="1567" height="631" alt="image" src="https://github.com/user-attachments/assets/99b30167-248b-47fe-ba79-ac4039163087" />
<img width="1602" height="539" alt="image" src="https://github.com/user-attachments/assets/82a2742e-9005-4d2d-86fb-1bda4db987c4" />

## Observation
Logistic Regression successfully classified most samples correctly.
Standardizing features improved model performance.
Gradio interface allows easy testing with custom inputs.

## Result
This experiment demonstrates a working AI model for cancer prediction. Using Logistic Regression and Gradio, predictions can be made interactively or on predefined samples.
