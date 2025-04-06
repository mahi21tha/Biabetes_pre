from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    data = np.array(data).reshape(1, -1)
    scaled = scaler.transform(data)
    result = model.predict(scaled)
    return render_template("index.html", prediction=result[0])

if __name__ == "__main__":
    app.run(debug=True)
