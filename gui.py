from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the pre-trained model
model_path = "random_forest_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Create a Flask app
app = Flask(__name__)

# Define the home route
@app.route("/")
def home():
    return render_template("index.html")

# Define the prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract input values from the form
        age = int(request.form["age"])
        gender = int(request.form["gender"])
        height = float(request.form["height"])
        weight = float(request.form["weight"])
        ap_hi = int(request.form["ap_hi"])
        ap_lo = int(request.form["ap_lo"])
        cholesterol = int(request.form["cholesterol"])
        gluc = int(request.form["gluc"])
        smoke = int(request.form["smoke"])
        alco = int(request.form["alco"])
        active = int(request.form["active"])

        # Prepare the input data for the model
        input_data = np.array([[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])

        # Make a prediction
        prediction = model.predict(input_data)[0]
        result = "High Risk" if prediction == 1 else "Low Risk"

        return render_template("index.html", prediction_text=f"Heart Attack Risk: {result}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)

@app.errorhandler(Exception)
def handle_exception(e):
    return f"Error: {str(e)}", 500

