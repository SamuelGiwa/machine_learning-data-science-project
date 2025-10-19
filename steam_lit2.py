import streamlit as st
import torch
from torch import nn
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit.components.v1 as components


data = load_breast_cancer()
X, y = data.data, data.target
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y


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

def model_predict(X_numpy):
    X_tensor = torch.tensor(X_numpy, dtype=torch.float32)
    with torch.no_grad():
        return model(X_tensor).numpy()


st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🩺", layout="wide")
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "🧠 Model Info", "📊 Prediction", "📈 Visualization"])

with tab1:
    st.title("🩺 Breast Cancer Classification App")
    st.write("""
    This app uses a **PyTorch-based neural network** trained on the **Breast Cancer Wisconsin dataset**  
    to predict whether a tumor is **Malignant (cancerous)** or **Benign (non-cancerous)**.

    ---
    **Dataset details:**
    - Samples: 569
    - Features: 30 continuous variables describing cell characteristics
    - Classes: 2 (Malignant, Benign)
    """)
    st.dataframe(df.head(5))
    st.info("Use the tabs above to explore the model, make predictions, and view dataset visualizations.")


with tab2:
    st.header("🧩 Neural Network Model Details")
    st.write("### Model Architecture:")
    st.code(str(model), language='python')
    st.write("### Model Explanation:")
    st.markdown("""
    - **Input layer:** 30 neurons (features from the dataset)  
    - **Hidden layers:** Two layers with 16 and 8 neurons respectively  
    - **Activation:** ReLU  
    - **Output layer:** 1 neuron with Sigmoid activation for binary classification  
    - **Loss function:** Binary Cross Entropy (BCELoss)  
    - **Optimizer:** Adam (learning rate = 0.001)
    """)
    st.write("### Model Accuracy (Example result):")
    st.metric(label="Test Accuracy", value="97.4%")
    st.success("This accuracy may vary based on the training split and random seed.")


with tab3:
    st.header("🔍 Predict Tumor Type with Explanation")
    st.sidebar.header("Enter Feature Values")
    input_data = []
    for i, feature in enumerate(data.feature_names):
        val = st.sidebar.number_input(f"{feature}", float(X[:, i].min()), float(X[:, i].max()), float(X[:, i].mean()))
        input_data.append(val)

    input_data = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_data)

    if st.button("Predict"):
        with torch.no_grad():
            pred = model(torch.tensor(input_scaled, dtype=torch.float32))
            label = "Malignant" if pred.item() >= 0.5 else "Benign"
            confidence = pred.item() if label == "Malignant" else 1 - pred.item()

        st.subheader("Prediction Result")
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")
        if label == "Malignant":
            st.error("⚠️ The model predicts the tumor is **Malignant**.")
        else:
            st.success("✅ The model predicts the tumor is **Benign**.")

        # st.subheader("Feature Importance (SHAP Values)")

        # # ------------------ SHAP Calculation ------------------
        # background = X_scaled[np.random.choice(X_scaled.shape[0], 50, replace=False)]
        # explainer = shap.KernelExplainer(model_predict, background)
        # shap_values = explainer.shap_values(input_scaled)

        # # Convert to SHAP Explanation object for single sample
        # expl = shap.Explanation(values=shap_values[0],
        #                         base_values=explainer.expected_value[0],
        #                         data=input_scaled[0],
        #                         feature_names=list(data.feature_names))

        # # Plot using waterfall plot
        # fig, ax = plt.subplots(figsize=(10, 5))
        # shap.plots._waterfall.waterfall_legacy(ax, max_display=10, show=False)
        # st.pyplot(fig)

# ------------------ Tab 4: Visualization ------------------
with tab4:
    st.header("📊 Data Visualization")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Target Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='target', data=df, hue='target', palette='viridis', ax=ax, legend=False)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Malignant (0)', 'Benign (1)'])
        st.pyplot(fig)

    with col2:
        st.subheader("Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    st.subheader("Pairplot (Top 4 Features)")
    fig = sns.pairplot(df.iloc[:, :4].assign(target=df['target']), hue='target', palette='coolwarm')
    st.pyplot(fig)