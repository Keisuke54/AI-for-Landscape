import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
from keras.models import load_model
from PIL import Image
from io import BytesIO
import requests

response = requests.get('https://deepstackpython.readthedocs.io/en/latest/_images/test-image3.jpg')
image = Image.open(BytesIO(response.content))
image = np.asarray(image)

fig1 = plt.figure("Figure 1")
plt.imshow(image)
plt.axis('off')
plt.show(block=False)

print('1')

category = ['3x3','Centered','Diagonal','Framing','SC-Shape','Split','Symmetry','Vanishing']

with tf.device('/GPU:0'):

    newModel = load_model('landModel.h5', compile=False)
    newModel.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    prediction = newModel.predict(np.expand_dims(image, axis=0))
    predLabel = np.argmax(prediction)
    
    prediction1 = prediction.reshape(-1)
    fig2 = plt.figure("Figure 2")
    plt.bar(category, prediction1)
    plt.xticks(rotation=45, ha='right')
    plt.title('Composition Prediction')
    plt.xlabel('Prediction')
    plt.ylabel('Score')
    plt.show(block=False)

print("Predicted label: " + category[predLabel])
print('end')
