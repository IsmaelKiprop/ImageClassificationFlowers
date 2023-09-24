import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load your trained model
model = load_model('/mnt/c/Users/ISMAEL/OneDrive/Desktop/ImageClassificationFlowers/models/my_model.h5')  # Replace with your model's path

# Define a function to load and preprocess an image for prediction
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Specify the path to the image you want to classify
image_path = '/mnt/c/Users/ISMAEL/OneDrive/Desktop/ImageClassificationFlowers/data/train/rose/image2.jpg' # Replace with the image you want to classify

# Use your model to predict the class label for the image
input_image = load_and_preprocess_image(image_path)
predictions = model.predict(input_image)
predicted_class_index = np.argmax(predictions)

# Define a dictionary that maps class indices to class labels
class_labels = {0: 'daisy', 1: 'rose'}  # Modify this based on your class labels

# Get the predicted class label
predicted_class_label = class_labels[predicted_class_index]

# Create the "visualizations" directory if it doesn't exist
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# Save the input image with the predicted class label in the "visualizations" directory
output_image_path = os.path.join('visualizations', f'predicted_{predicted_class_label}.jpg')

# Visualize the input image and its prediction
plt.figure(figsize=(8, 8))
plt.imshow(cv2.imread(image_path))
plt.axis('off')
plt.title(f'Predicted Class: {predicted_class_label}')

# Save the visualization
plt.savefig(output_image_path)
plt.show()

print(f'Visualization saved at {output_image_path}')
