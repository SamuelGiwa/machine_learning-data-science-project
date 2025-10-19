import streamlit as st
import torch
from torch import nn
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import numpy as np


data = load_breast_cancer()
X, y = data.data, data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

class BreastCancerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(30, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)


model = BreastCancerModel()
model.eval()


st.title("🩺 Breast Cancer Prediction App")
st.write("This app uses a neural network trained on the Breast Cancer dataset to predict whether a tumor is **Malignant** or **Benign**.")


st.sidebar.header("Input Features")
input_data = []

for i, feature in enumerate(data.feature_names):
    val = st.sidebar.number_input(f"{feature}", float(X[:, i].min()), float(X[:, i].max()), float(X[:, i].mean()))
    input_data.append(val)


input_data = np.array(input_data).reshape(1, -1)
input_scaled = scaler.transform(input_data)
input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

if st.button("Predict"):
    with torch.no_grad():
        pred = model(input_tensor)
        label = "Malignant" if pred.item() >= 0.5 else "Benign"
        confidence = pred.item() if label == "Malignant" else 1 - pred.item()

    st.subheader("Prediction Result")
    st.write(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")

    if label == "Malignant":
        st.error("⚠️ The model predicts the tumor is Malignant.")
    else:
        st.success("✅ The model predicts the tumor is Benign.")
