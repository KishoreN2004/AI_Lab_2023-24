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

# Step 7: Prediction function with empty input handling
def lung_cancer_prediction(Age, Smoking, Alcohol, Obesity, Family_History):
    try:
        x_input = np.array([
            Age if Age is not None else 0,
            Smoking if Smoking is not None else 0,
            Alcohol if Alcohol is not None else 0,
            Obesity if Obesity is not None else 0,
            Family_History if Family_History is not None else 0
        ]).reshape(1, -1)
        # Scale the input
        x_input_scaled = scaler.transform(x_input)
        pred = model.predict(x_input_scaled)
        return "YES" if pred[0] == 1 else "NO"
    except:
        return "Invalid Input"

# Step 8: Predefined sample patients
sample_patients = [
    {"Age":45,"Smoking":1,"Alcohol":1,"Obesity":0,"Family_History":1},
    {"Age":30,"Smoking":0,"Alcohol":0,"Obesity":0,"Family_History":0},
    {"Age":65,"Smoking":1,"Alcohol":1,"Obesity":1,"Family_History":1},
    {"Age":28,"Smoking":0,"Alcohol":0,"Obesity":0,"Family_History":0},
    {"Age":55,"Smoking":0,"Alcohol":0,"Obesity":0,"Family_History":0},
    {"Age":70,"Smoking":1,"Alcohol":1,"Obesity":1,"Family_History":1},
    {"Age":None,"Smoking":None,"Alcohol":None,"Obesity":None,"Family_History":None},  # empty sample
]

# Step 9: Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown(f"## Lung Cancer Prediction - Preloaded Samples\n**Model Accuracy:** {model_accuracy*100:.2f}%")

    # Create input components for each sample
    input_components = []
    output_boxes = []

    for i, patient in enumerate(sample_patients):
        with gr.Row():
            age = gr.Number(value=patient["Age"], label=f"Sample {i+1} - Age")
            smoking = gr.Number(value=patient["Smoking"], label=f"Sample {i+1} - Smoking")
            alcohol = gr.Number(value=patient["Alcohol"], label=f"Sample {i+1} - Alcohol")
            obesity = gr.Number(value=patient["Obesity"], label=f"Sample {i+1} - Obesity")
            family = gr.Number(value=patient["Family_History"], label=f"Sample {i+1} - Family History")
            out = gr.Textbox(label=f"Prediction Sample {i+1}")
            input_components.append((age, smoking, alcohol, obesity, family))
            output_boxes.append(out)

    def predict_all(*args):
        results = []
        for i in range(len(sample_patients)):
            Age, Smoking, Alcohol, Obesity, Family_History = args[i*5:(i+1)*5]
            result = lung_cancer_prediction(Age, Smoking, Alcohol, Obesity, Family_History)
            results.append(result)
        return results

    btn = gr.Button("Predict All Samples")
    btn.click(predict_all, inputs=[comp for tup in input_components for comp in tup], outputs=output_boxes)

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
