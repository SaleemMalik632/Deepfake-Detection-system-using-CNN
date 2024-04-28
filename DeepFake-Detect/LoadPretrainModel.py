import os
import pandas as pd
from tensorflow.keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Dropout
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf


dataset_path = 'D:\Github\A-6th\CVIP\Project\DeepFake-Detect'
# Define the custom FixedDropout layer
class FixedDropout(Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = tf.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)
get_custom_objects().update({'FixedDropout': FixedDropout})
checkpoint_filepath = './tmp_checkpoint'
model = load_model(os.path.join(checkpoint_filepath, 'best_model.h5'))

test_datagen = ImageDataGenerator(rescale=1/255)

test_generator = test_datagen.flow_from_directory(
    directory=os.path.join(dataset_path, 'prepared_dataset'),
    classes=['real', 'fake'],
    target_size=(128, 128),
    color_mode="rgb",
    class_mode=None,
    batch_size=4,
    shuffle=False
) 
test_generator.reset()
preds = model.predict(
    test_generator,
    verbose = 1
)
test_results = pd.DataFrame({
    "Filename": test_generator.filenames,
    "Prediction": preds.flatten()
}) 
test_results['Label'] = test_results['Prediction'].apply(lambda x: 'Real' if x > 0.5 else 'Fake')
print('Result After pridiction = ',test_results)
test_results.to_csv('pre.csv', index=False)
print('Results saved to predictions.csv')






def predict_image(model, image_path, input_size=128):
    image = load_img(image_path, target_size=(input_size, input_size))
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image)
    label = 'Real' if pred > 0.5 else 'Fake'
    return label

image_path = './fake.png'
print('Prediction:', predict_image(model, image_path))