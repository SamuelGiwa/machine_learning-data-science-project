import joblib
import numpy as np
import os

def test_model_prediction():
    model_path = os.path.join("api", "model", "linear_regression_model.pkl")
    model = joblib.load(model_path)

    
    sample_weight = np.array([[70]])  
    prediction = model.predict(sample_weight)

    assert prediction.shape == (1,)
    assert prediction[0] > 0 and prediction[0] < 250  
