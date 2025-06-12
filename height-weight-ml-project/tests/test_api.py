from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_predict_height():
    response = client.post("/predict_height", json={"weight": 70})
    
    assert response.status_code == 200
    result = response.json()
    
    assert "height" in result
    assert type(result["height"]) in [int, float]
    assert 50 < result["height"] < 250  