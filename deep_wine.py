import platform
import os

if (platform.system() == 'Windows'):
    os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
    
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, GlobalAveragePooling1D
from keras.utils import plot_model, to_categorical
from keras.models import load_model
from keras.preprocessing.text import text_to_word_sequence, one_hot, hashing_trick
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("wine_ml.csv", ",")

train, validate, test = np.split(data.sample(frac=1), [int(.6*len(data)), int(.8*len(data))])

print(len(train))
print(len(validate))
print(len(test))

vocab_size = train['text'].nunique()
classes = train['taster_name'].nunique()
print(vocab_size)
print(classes)

def rowFunc(row):
    return one_hot(row, vocab_size)

x_train, y_train = train['text'].apply(rowFunc).values, train['taster_name'].apply(one_hot, n=classes, split='pleasedont').values
y_train = [item for sublist in y_train for item in sublist]
y_train = to_categorical(y_train)

# Tokenize the strings
train_str = train['text']
validate_str = validate['text'].to_string
test_str = test['text'].to_string
words_train = text_to_word_sequence(train_str)
words_val = text_to_word_sequence(validate_str)
words_test = text_to_word_sequence(test_str)

train_vocab_size = len(words_train)

x_train = one_hot(train['text'], round(train_vocab_size*1.3))

network = Sequential([
    Dense(30), 
    Activation('relu'),
    Dense(60), 
    Activation('relu'),
    Dense(90), 
    Activation('relu'),
    Dense(120), 
    Activation('relu'),
    Dense(48), 
    Activation('sigmoid'),
    Dense(19),
    Activation('softmax')
    ])

network.compile(
    optimizer='adam',
    loss='categorical_crossentropy', # Categorical classification
    metrics=['accuracy']
    )

network.fit(
    x_train, y_train,
    epochs=10,
    batch_size=56
    )