from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model/breast_cancer_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            features = [
                float(request.form["radius_mean"]),
                float(request.form["texture_mean"]),
                float(request.form["perimeter_mean"]),
                float(request.form["area_mean"]),
                float(request.form["smoothness_mean"])
            ]

            features = np.array(features).reshape(1, -1)
            features_scaled = scaler.transform(features)
            result = model.predict(features_scaled)[0]

            prediction = "Malignant" if result == 1 else "Benign"

        except:
            prediction = "Invalid input. Please enter numeric values."

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
