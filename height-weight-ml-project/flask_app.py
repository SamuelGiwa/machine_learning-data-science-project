from flask import Flask, request, render_template , jsonify
import joblib



model = joblib.load('linear_regression_model.pkl')


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()   
    weight = float(data['weight'])
    height_pred = model.predict([[weight]])[0]
    return jsonify({'predicted_height': round(height_pred, 2), 'input_weight': weight})


if __name__ == '__main__':
    app.run(debug=True)