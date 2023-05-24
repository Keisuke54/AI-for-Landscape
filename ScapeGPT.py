# pip install ultralytics 
# pip install gTTS

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
from keras.models import load_model
import cv2
import requests
 
# insert your image link here 
# Example
response = requests.get('https://deepstackpython.readthedocs.io/en/latest/_images/test-image3.jpg')
#response = requests.get('https://wallpaperaccess.com/full/4933291.jpg')
image = Image.open(BytesIO(response.content))
image = np.asarray(image)

# Detection
model = YOLO("yolov8n.pt")
yoloLabels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

predict = model.predict(image, conf=0.25)
results = predict[0].boxes.data

detected = []

fig1 = plt.figure("Figure 1")
for result in results:
    x1 = int(result[0])
    y1 = int(result[1])
    x2 = int(result[2])
    y2 = int(result[3])
    confidence = result[4]
    label = yoloLabels[int(result[5])]

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"{label}: {confidence:.2f}"
    cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if label not in detected:
        detected.append(label)

plt.imshow(image)
plt.axis('off')
plt.show(block = False)

detResult = ''
z = 1
count = 0
for object in detected:
    count = detected.count(object)
    if z == len(detected):
        if count == 1:
            detResult += str(count) + ' ' + object
        else: 
            detResult += str(count) + ' ' + object
    else:
        if count == 1:
            detResult += str(count) + ' ' + object + ', '
        else: 
            detResult += str(count) + ' ' + object + 's, '
    z += 1
    


# Classification 
category = ['3x3','Centered','Diagonal','Framing','SC-Shape','Split','Symmetry','Vanishing']

with tf.device('/GPU:0'):
    landModel = load_model('landModel.h5', compile=False)
    landModel.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    pred = landModel.predict(np.expand_dims(image, axis=0))
    predLabel = np.argmax(pred)
    pred1 = pred.reshape(-1)

fig2 = plt.figure("Figure 2")
plt.bar(category, pred1)
plt.xticks(rotation=45, ha='right')
plt.title('Composition Prediction')
plt.xlabel('Prediction')
plt.ylabel('Score')

prompt = 'How can I draw ' + category[predLabel] + ' composition landscape with ' + detResult + '?'

print(' ')
print('Question:' + prompt)
print(' ')


# LLM
from conversation import conversation 

with tf.device('/CPU:0'):
    Tokenizer = tf.keras.preprocessing.text.Tokenizer
    
    pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
    
    tokenizer = Tokenizer(char_level=True, lower=True)
    tokenizer.fit_on_texts(conversation)
    
    def generate_text(seed_text, model, tokenizer, sequence_length, num_chars_to_generate):
        generated_text = seed_text
        removed_text = ''
        
        for _ in range(num_chars_to_generate):
            token_list = tokenizer.texts_to_sequences([generated_text])
            token_list = pad_sequences(token_list, maxlen=sequence_length, padding="pre")
            predicted_probs = model.predict(token_list, verbose=0)
            predicted_token = np.argmax(predicted_probs, axis=-1)[0]  
            
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted_token:
                    output_word = word
                    break
            
            generated_text += output_word
            removed_text += output_word

        return removed_text
    
    modelF = load_model('Lllm.h5', compile=False)
    modelF.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    generated_text = generate_text(prompt, modelF, tokenizer, 100, num_chars_to_generate=400)

print(' ')
print('Anser:' + generated_text)
print(' ')
plt.show()