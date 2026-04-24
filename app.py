import gradio as gr
import pandas as pd
import pickle 
import numpy as np
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
def predict(gender, age, estimated_salary):
    input_df=pd.DataFrame([[gender, age, estimated_salary]], columns=["Gender", "Age", "EstimatedSalary"])
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        return "Likely to Purchase"
    else:
        return "Unlikely to Purchase"
inputs=[
    gr.Dropdown(choices=["Male", "Female"], label="Gender"),
    gr.Slider(minimum=18, maximum=60, step=1, label="Age"),
    gr.Slider(minimum=15000, maximum=150000, step=1000, label="Estimated Salary")
]

app=gr.Interface(fn=predict, inputs=inputs, outputs='text', title="Purchase Prediction")
app.launch()
