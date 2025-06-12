# Height-Weight ML Project

This project builds a machine learning pipeline to predict height from weight.  
It includes data analysis, model training, an API server, and an interactive app.

---

## Project Structure

height-weight-ml-project/
│
├── data/
│ ├── height_weight.csv # Original dataset
│ ├── weights.txt # List of weights to predict on
│ ├── height_predictions.csv # Output predictions
│
├── test/
|   ├── test_api.py
|   ├── test_model.py
|
|
|
├── notebooks/
│ └── eda_and_modeling.ipynb # EDA and model training notebook
│
├── app.py # Interactive app (Streamlit or Dash)
│
├── api/
│ ├── init.py
│ ├── main.py # FastAPI app exposing /predict_height
│ ├── model/
│ │ ├── linear_regression_model.pkl
│
├── scripts/
│ ├── call_api_loop.sh # Script to batch-call the API
│
├── Dockerfile # Docker config
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore

---

## Features

✅ Data Exploration and Visualization  
✅ Linear Regression Model  
✅ FastAPI-based prediction API  
✅ Streamlit/Dash interactive app  
✅ Automation script to call API in batch  
✅ Docker-ready  

---

## Setup Instructions

1. **Clone the repository**  

    ```bash
    git clone <your-repo-url>
    cd height-weight-ml-project
    ```

2. **Install dependencies**  
    (You may want to use a virtual environment.)  

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the FastAPI server**

    ```bash
    uvicorn api.main:app --reload
    ```

4. **Run the Streamlit/Dash app**  

    ```bash
    streamlit run app.py
    ```

5. **Call API in batch**  

    ```bash
    bash scripts/call_api_loop.sh
    ```

---

## API Endpoint

**`POST /predict_height`**

- **Input**: JSON with `weight` field  
  Example:
  
  ```json
  { "weight": 75 }


---

What I Learned
End-to-end ML pipeline design

FastAPI + Streamlit integration

Model deployment best practices

Docker for reproducibility



