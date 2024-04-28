import json
import os
from distutils.dir_util import copy_tree
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K 
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from efficientnet.tfkeras import EfficientNetB0

# Set TensorFlow version
print('TensorFlow version: ', tf.__version__)

# Set dataset paths
# dataset_path = 'D:\Github\A-6th\CVIP\Project\DeepFake-Detect\github data\split_dataset'
dataset_path = 'D:/Github/A-6th/CVIP/Project/DeepFake-Detect/split_dataset'
tmp_debug_path = './tmp_debug'
checkpoint_filepath = './tmp_checkpoint'

# Create directories if they don't exist
os.makedirs(tmp_debug_path, exist_ok=True)
os.makedirs(checkpoint_filepath, exist_ok=True)

# Function to extract filename without extension
def get_filename_only(file_path):
    file_basename = os.path.basename(file_path) 
    filename_only = file_basename.split('.')[0]
    return filename_only

# Set parameters
input_size = 128
batch_size_num = 32
num_epochs = 20

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    directory= os.path.join('D:\Github\A-6th\CVIP\Project\DeepFake-Detect\github data\split_dataset', 'train'),
    target_size=(input_size, input_size),
    color_mode="rgb",
    class_mode="binary",
    batch_size=batch_size_num,
    shuffle=True,
    save_to_dir=tmp_debug_path
)

val_generator = val_datagen.flow_from_directory(
    directory= os.path.join('D:\Github\A-6th\CVIP\Project\DeepFake-Detect\github data\split_dataset', 'val'),
    target_size=(input_size, input_size),
    color_mode="rgb",
    class_mode="binary",
    batch_size=batch_size_num,
    shuffle=True,
    save_to_dir=tmp_debug_path
)

test_generator = test_datagen.flow_from_directory(
    directory=os.path.join('D:\Github\A-6th\CVIP\Project\DeepFake-Detect\github data\split_dataset', 'test'),
    classes=['real', 'fake'],
    target_size=(input_size, input_size),
    color_mode="rgb",
    class_mode=None,
    batch_size=1,
    shuffle=False
)

# Base model
efficient_net = EfficientNetB0(
    weights='imagenet',
    input_shape=(input_size, input_size, 3),
    include_top=False,
    pooling='max'
)

# Model architecture
model = Sequential()
model.add(efficient_net)
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compile model
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks that will be used during training to stop training early if the model is not improving and to save the best model
custom_callbacks = [
    EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=5,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(checkpoint_filepath, 'best_model.h5'),
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_best_only=True
    )
]

# Train model by fitting the generator to the model 
history = model.fit_generator(
    train_generator,
    epochs=num_epochs,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=custom_callbacks,
)

# Load best model
best_model = load_model(os.path.join(checkpoint_filepath, 'best_model.h5'))

# Generate predictions
test_generator.reset()
preds = best_model.predict(
    test_generator,
    verbose = 1
)
test_results = pd.DataFrame({
    "Filename": test_generator.filenames,
    "Prediction": preds.flatten()
}) 

# Add 'Label' column
test_results['Label'] = test_results['Prediction'].apply(lambda x: 'Real' if x > 0.5 else 'Fake')

print('Result After pridiction = ',test_results)
# Save the results to a CSV file
test_results.to_csv('predictions1.csv', index=False)

print('Results saved to predictions.csv')