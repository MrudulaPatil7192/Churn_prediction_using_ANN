import os
import numpy as np
from flask import Flask, render_template_string, request, jsonify

# Force TensorFlow / Keras to use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import keras

app = Flask(__name__)

# Load the saved Keras ANN model[cite: 1]
MODEL_PATH = "ANN.pkl"
model = keras.models.load_model(MODEL_PATH)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Churn Prediction System</title>
    <style>
        * { box-sizing: border-box; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        body { background: #0f172a; color: #f8fafc; margin: 0; padding: 40px 20px; display: flex; justify-content: center; min-height: 100vh; align-items: center; }
        .card { background: #1e293b; padding: 32px; border-radius: 16px; max-width: 650px; width: 100%; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5); border: 1px solid #334155; }
        h2 { margin-top: 0; color: #f1f5f9; border-bottom: 2px solid #334155; padding-bottom: 12px; font-size: 24px; text-align: center; }
        .grid-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 16px; margin-top: 24px; }
        .input-group { display: flex; flex-direction: column; }
        .input-group label { font-size: 12px; font-weight: 600; color: #94a3b8; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; }
        .input-group input { padding: 10px 12px; background: #0f172a; border: 1px solid #334155; border-radius: 8px; font-size: 14px; color: #f8fafc; outline: none; transition: border-color 0.2s; }
        .input-group input:focus { border-color: #38bdf8; }
        button { margin-top: 28px; width: 100%; padding: 14px; background: #2563eb; color: white; border: none; border-radius: 8px; font-size: 16px; font-weight: 600; cursor: pointer; transition: background 0.2s; }
        button:hover { background: #1d4ed8; }
        .result-box { margin-top: 24px; padding: 16px; border-radius: 8px; display: none; text-align: center; font-size: 16px; font-weight: 600; }
        .success { background: #064e3b; color: #6ee7b7; border: 1px solid #047857; }
        .error { background: #7f1d1d; color: #fca5a5; border: 1px solid #b91c1c; }
    </style>
</head>
<body>

<div class="card">
    <h2>Customer Churn Predictor</h2>
    <form id="predictionForm">
        <div class="grid-container">
            <script>
                for (let i = 1; i <= 10; i++) {
                    document.write(`
                        <div class="input-group">
                            <label for="f${i}">Feature ${i}</label>
                            <input type="number" step="any" id="f${i}" name="f${i}" placeholder="0.0" required>
                        </div>
                    `);
                }
            </script>
        </div>
        <button type="submit">Predict Churn</button>
    </form>

    <div id="resultBox" class="result-box"></div>
</div>

<script>
    document.getElementById("predictionForm").addEventListener("submit", async function(e) {
        e.preventDefault();
        
        const features = [];
        for (let i = 1; i <= 10; i++) {
            features.push(parseFloat(document.getElementById(`f${i}`).value));
        }

        const resultBox = document.getElementById("resultBox");
        resultBox.style.display = "block";
        resultBox.className = "result-box";
        resultBox.innerHTML = "Processing neural network output...";

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ features: features })
            });

            const data = await response.json();

            if (response.ok) {
                resultBox.className = "result-box success";
                const label = data.predicted_class === 1 ? "Likely to Churn" : "Unlikely to Churn";
                resultBox.innerHTML = `Prediction: <b>${label} (Class ${data.predicted_class})</b><br><small>Confidence: ${(data.probability * 100).toFixed(2)}%</small>`;
            } else {
                resultBox.className = "result-box error";
                resultBox.innerHTML = `Error: ${data.error}`;
            }
        } catch (err) {
            resultBox.className = "result-box error";
            resultBox.innerHTML = "Unable to communicate with the server.";
        }
    });
</script>

</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data.get("features", [])
        
        if len(features) != 10:
            return jsonify({"error": f"Expected 10 inputs, received {len(features)}"}), 400

        input_data = np.array([features], dtype=np.float32)
        prediction = model.predict(input_data)[0][0]
        class_result = 1 if prediction >= 0.5 else 0

        return jsonify({
            "probability": float(prediction),
            "predicted_class": class_result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Fetch port dynamically for Render hosting deployment
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
