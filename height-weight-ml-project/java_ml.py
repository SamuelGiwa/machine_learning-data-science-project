from sklearn.linear_model import LinearRegression
from sklearn2pmml import sklearn2pmml, PMMLPipeline
import pandas as pd

data = pd.read_csv('height-weight.csv')
X = data[['Weight']]
y = data['Height']

pipeline = PMMLPipeline([
    ("regressor", LinearRegression())
])
pipeline.fit(X, y)

# Export to PMML file
sklearn2pmml(pipeline, "linear_regression.pmml", with_repr=True)