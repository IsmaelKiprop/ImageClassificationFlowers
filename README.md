# Flower Image Classification

![image](https://github.com/IsmaelKiprop/ImageClassificationFlowers/assets/133222922/c8f8a8f3-1da0-48dc-864f-4520668044ff)


## Overview

This project focuses on training a machine learning model to classify images of two different types of flowers: roses and daisies. The goal is to create a model that can accurately predict the type of flower present in a given image.

## Key Components

1. **Data Collection**: The dataset contains a collection of images, with examples of both roses and daisies.

2. **Data Preprocessing**: Images are preprocessed to ensure uniformity and to prepare them for training. This includes resizing, normalizing pixel values, and formatting.

3. **Model Selection**: A pre-trained deep learning model, ResNet50, is used as the base model. Custom layers are added for fine-tuning.

4. **Model Fine-Tuning**: Custom layers are added to adapt the base model for the specific classification task.

5. **Training**: The model is trained using the preprocessed dataset. Hyperparameters like epochs and batch size are specified.

6. **Evaluation**: Model performance is evaluated using a separate test dataset, and metrics are calculated.

7. **Visualization**: Visualizations are generated to help understand the model's predictions. These include input images and predicted class labels.

8. **Prediction**: A script for making predictions on new images is provided. Users can input an image, and the model predicts whether it's a rose or a daisy.

   ![image](https://github.com/IsmaelKiprop/ImageClassificationFlowers/assets/133222922/0bcee512-9a65-4e0f-ba7c-bf1abbd5abf6)


## Usage

To use this project:

1. Clone the repository:

   ```bash
   git clone https://github.com/IsmaelKiprop/ImageClassificationFlowers.git

2. Install the required libraries:

pip install -r requirements.txt

## Create a folder models

3. Train the model:

python train.py

## \models\my_model.h5 will be generated.

4. Make predictions:

python predict.py --image path/to/your/image.jpg

## Run visualize.py  to generate images predicting flower types

5. Results

Results and performance metrics are documented in the project. Further enhancements and optimizations can be explored to improve accuracy.

## Contributing
Contributions are welcome! If you have ideas for improvements or find issues, please create a GitHub issue or make a pull request.
