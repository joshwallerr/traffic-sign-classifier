from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model = load_model("traffic_ai_v2.h5")

def predict_sign(image_path):
    # Open and preprocess the image
    img = Image.open(image_path)
    img = img.resize((30, 30))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Make it (1, 30, 30, 3)
    
    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    
    return predicted_class

# Example usage
image_path = r"stopsign2.JPG"
print(predict_sign(image_path))
