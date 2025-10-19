# from fastapi import FastAPI
# import joblib
# from typing import Optional
# from pydantic import BaseModel


# model = joblib.load('linear_regression_model.pkl')

# app = FastAPI()

# class Item(BaseModel):
#     Weight: float



# @app.get("/")
# def read_root():
#     return {"message": "Hello, FastAPI!"}

# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: str = None):
    
#     height_pred = model.predict([[item_id]])[0]
#     return {"height_prediction": height_pred}



from fastapi import FastAPI
import joblib
from pydantic import BaseModel


model = joblib.load('linear_regression_model.pkl')


app = FastAPI()


class Item(BaseModel):
    Weight: float

# Root route
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.get("/predict_height/{weight}")
def predict_height(weight: float):
    height_pred = model.predict([[weight]])[0]
    return {"height_prediction": height_pred}

@app.post("/predict_height")
def predict_height_post(item: Item):
    height_pred = model.predict([[item.Weight]])[0]
    return {"height_prediction": height_pred}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fast_api_endpoint:app", host="127.0.0.1", port=8000, reload=True)