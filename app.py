from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("brain_mri_model.h5")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Define a route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read the uploaded image file
        img_bytes = file.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)

        # Preprocess the image (resize and normalize)
        img = cv2.resize(img, (128, 128))
        img = img.reshape(1, 128, 128, 1) / 255.0

        # Make prediction
        prediction = model.predict(img)[0][0]
        result = "Tumor" if prediction > 0.5 else "No Tumor"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the app on port 5000
    app.run(host="0.0.0.0", port=5000, debug=True)
