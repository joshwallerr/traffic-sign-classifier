import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image

# Constants
EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

def load_data(data_dir):
    images = []
    labels = []

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                filepath = os.path.join(folder_path, filename)
                image = Image.open(filepath)
                image = image.resize((IMG_WIDTH, IMG_HEIGHT))
                images.append(np.array(image))
                labels.append(int(folder))
                print(f"Loaded {filepath}")

    return (images, labels)

def preprocess_data(images, labels):
    images = np.array(images) / 255.0
    labels = to_categorical(labels)
    return (images, labels)

def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CATEGORIES, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load the data
print("Loading data...")
data_dir = "gtsrb" # Update the path accordingly
images, labels = load_data(data_dir)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    images, labels, test_size=TEST_SIZE, stratify=labels
)

# Preprocess the data
x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)

# Build and train the model
model = build_model()
model.summary()
model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc}")

model.save("traffic_ai_v2.h5")