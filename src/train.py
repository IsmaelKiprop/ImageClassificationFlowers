import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Absolute paths to your data directories
train_data_dir = '/mnt/c/Users/ISMAEL/OneDrive/Desktop/ImageClassificationFlowers/data/train'
test_data_dir = '/mnt/c/Users/ISMAEL/OneDrive/Desktop/ImageClassificationFlowers/data/test'

# Parameters
batch_size = 32
num_epochs = 10
num_classes = len(os.listdir(train_data_dir))

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

# Create the base model (ResNet50)
base_model = ResNet50(weights='imagenet', include_top=False)

# Add custom layers for your task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=num_epochs,
    validation_data=test_generator,
    validation_steps=len(test_generator)
)

# Save the model with an absolute path
model_save_path = '/mnt/c/Users/ISMAEL/OneDrive/Desktop/ImageClassificationFlowers/models/my_model.h5'
model.save(model_save_path)