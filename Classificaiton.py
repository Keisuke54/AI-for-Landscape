# Work was done in Kaggle: https://www.kaggle.com/code/keisukenakamura54/composition-classificaiton-modified
# Use GPU or TPU

import numpy as np
import tensorflow as tf
import random
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator 

# preprocessing each image for Resnet
imageData = ImageDataGenerator(preprocessing_function=preprocess_input)

# generating training and validation sets 
image_size = 224
trainData = imageData.flow_from_directory(
        '../input/landscape-composition/Train',
        target_size=(image_size, image_size),
        class_mode='categorical')

valData = imageData.flow_from_directory(
        '../input/landscape-composition/Test',
        target_size=(image_size, image_size),
        class_mode='categorical')

from tensorflow.keras.applications import ResNet50
from tensorflow import keras 
from tensorflow.keras import layers

num_classes = 8

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

newModel = keras.Sequential(
    [   
        ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path),
        
        layers.Dense(num_classes, activation='softmax'),
    ]
)

newModel.layers[0].trainable = False

newModel.summary()

newModel.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

newModel.fit(
    trainData,
    validation_data=valData,
    epochs=10, 
)

newModel.save('/kaggle/working/landModel.h5')
newModel.save_weights('/kaggle/working/landWeights.h5')