# code run in Kaggle: https://www.kaggle.com/keisukenakamura54/llm-for-landscape 

import numpy as np
import tensorflow as tf

from conversation import conversation

Sequential = tf.keras.models.Sequential

Embedding = tf.keras.layers.Embedding
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense

Tokenizer = tf.keras.preprocessing.text.Tokenizer

pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
tokenizer = Tokenizer(char_level=True, lower=True)

tokenizer.fit_on_texts(conversation)
vocab_size = len(tokenizer.word_index) + 1

# Prepare input and target sequences
input_sequences = []
output_sequences = []

for x in range(len(conversation)):
  sequences = tokenizer.texts_to_sequences(conversation)[x]

  sequence_length = 100

  for i in range(len(sequences) - sequence_length):
    input_sequences.append(sequences[i:i + sequence_length])
    output_sequences.append(sequences[i + sequence_length])

input_sequences = np.array(input_sequences)
output_sequences = np.array(output_sequences)

# model architecture 
model = Sequential([
    Embedding(vocab_size, 32, input_length=sequence_length),
    LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(vocab_size, activation="softmax"),
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

epochs = 100
batch_size = 32
model.fit(input_sequences, output_sequences, epochs=epochs, batch_size=batch_size)

print('end')

model.save('/kaggle/working/Lllm.h5')
model.save_weights('/kaggle/working/LllmMWeights.h5')