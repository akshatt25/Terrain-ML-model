from flask import Flask, request, jsonify
from PIL import Image
import requests
from io import BytesIO
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Load the trained model for terrain prediction
model = tf.keras.models.load_model(r"C:\Users\HP\Downloads\terrain_recognition_model.h5")

@app.route('/predict_terrain', methods=['GET'])
def predict_terrain():
    try:
        # Get the image URL from the query parameters
        image_url = request.args.get('image_url')

        if not image_url:
            return jsonify({"error": "Missing 'image_url' query parameter"}), 400

        # Fetch the image from the URL
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))

        # Preprocess the image for prediction
        img = image.resize((224, 224))  # Resize to the model's input shape
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make a prediction using the trained model
        prediction = model.predict(img_array)
        terrain_labels = ['grassy', 'marshy', 'rocky', 'sandy', 'snowy']
        predicted_class = np.argmax(prediction)
        predicted_terrain = terrain_labels[predicted_class]

        return jsonify({"predicted_terrain": predicted_terrain})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082)
