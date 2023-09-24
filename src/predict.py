import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('/mnt/c/Users/ISMAEL/OneDrive/Desktop/ImageClassificationFlowers/models/my_model.h5')


# Define the path to the image you want to classify
image_path = '/mnt/c/Users/ISMAEL/OneDrive/Desktop/ImageClassificationFlowers/data/train/rose/image2.jpg'

# Load and preprocess the image
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize the image

# Make predictions
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)

# Get class labels from directory names
class_labels = sorted(os.listdir('/mnt/c/Users/ISMAEL/OneDrive/Desktop/ImageClassificationFlowers/data/train'))

# Print the predicted class label
print(f'Predicted class: {class_labels[predicted_class]}')