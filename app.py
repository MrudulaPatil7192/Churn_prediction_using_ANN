from flask import Flask, render_template_string, request, jsonify
import keras
import numpy as np

app = Flask(__name__)

# Load the Keras ANN model saved in the .pkl file[cite: 1]
MODEL_PATH = "ANN.pkl"
model = keras.models.load_model(MODEL_PATH)

# Embedded HTML/CSS/JS Interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ANN Prediction Interface</title>
    <style>
        * { box-sizing: border-box; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        body { background: #f0f2f5; margin: 0; padding: 40px 20px; display: flex; justify-content: center; }
        .card { background: #ffffff; padding: 30px; border-radius: 12px; max-width: 650px; width: 100%; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
        h2 { margin-top: 0; color: #1a1a1a; border-bottom: 2px solid #eef2f5; padding-bottom: 12px; }
        .grid-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 15px; margin-top: 20px; }
        .input-group { display: flex; flex-direction: column; }
        .input-group label { font-size: 13px; font-weight: 600; color: #555; margin-bottom: 6px; }
        .input-group input { padding: 10px; border: 1px solid #ccc; border-radius: 6px; font-size: 14px; outline: none; transition: border-color 0.2s; }
        .input-group input:focus { border-color: #007bff; }
        button { margin-top: 25px; width: 100%; padding: 12px; background: #007bff; color: white; border: none; border-radius: 6px; font-size: 16px; font-weight: 600; cursor: pointer; transition: background 0.2s; }
        button:hover { background: #0056b3; }
        .result-box { margin-top: 25px; padding: 15px; border-radius: 8px; display: none; text-align: center; font-size: 16px; font-weight: bold; }
        .success { background: #e6f4ea; color: #137333; border: 1px solid #ceead6; }
        .error { background: #fce8e6; color: #c5221f; border: 1px solid #fad2cf; }
    </style>
</head>
<body>

<div class="card">
    <h2>ANN Model Predictor</h2>
    <form id="predictionForm">
        <div class="grid-container">
            <!-- Dynamically generate 10 feature input fields required by the model -->
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
        <button type="submit">Predict Result</button>
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
        resultBox.innerHTML = "Processing prediction...";

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ features: features })
            });

            const data = await response.json();

            if (response.ok) {
                resultBox.className = "result-box success";
                resultBox.innerHTML = `Predicted Class: <b>${data.predicted_class}</b><br><small>Confidence Probability: ${(data.probability * 100).toFixed(2)}%</small>`;
            } else {
                resultBox.className = "result-box error";
                resultBox.innerHTML = `Error: ${data.error}`;
            }
        } catch (err) {
            resultBox.className = "result-box error";
            resultBox.innerHTML = "An error occurred connecting to the server.";
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
        
        # Validate 10 input features
        features = data.get("features", [])
        if len(features) != 10:
            return jsonify({"error": f"Expected 10 input values, received {len(features)}"}), 400

        # Convert input array to shape (1, 10)
        input_data = np.array([features], dtype=np.float32)

        # Run model inference[cite: 1]
        prediction = model.predict(input_data)[0][0]
        
        # Classification threshold
        class_result = 1 if prediction >= 0.5 else 0

        return jsonify({
            "probability": float(prediction),
            "predicted_class": class_result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
