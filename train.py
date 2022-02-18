from data import create_generators
from model import *
from analyse import *
from sklearn.metrics import confusion_matrix
from platform import python_version_tuple

if python_version_tuple()[0] == '3':
    xrange = range
    izip = zip
    imap = map
else:
    from itertools import izip, imap

import numpy as np


def get_data(input_shape):
    train, val, test = create_generators()
    return train, val, test

def get_model(input_shape):
    x_input = tf.keras.layers.Input(input_shape)
    x_output = calculate_probability(x_input)
    return tf.keras.Model(inputs=x_input, outputs=x_output)

def train_model():
    train, val, test = get_data((*TRAINING_IMAGE_SIZE,NUMBER_OF_CHANNELS))
    model = get_model((*TRAINING_IMAGE_SIZE,NUMBER_OF_CHANNELS))
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=["accuracy"])
    model.fit(train,
              epochs=4,
              steps_per_epoch=94
                        )
    return test, model

test, model = train_model()
x,y=test[0]
cm = confusion_matrix(np.argmax(y,axis=1), np.argmax(model(x),axis=1), normalize='true')
for i in range(1,len(test)):
  x,y=test[i]
  cm += confusion_matrix(np.argmax(y,axis=1), np.argmax(model(x),axis=1), normalize='true')

cm = cm / len(test)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def show_confusion_matrix(matrix, labels):
    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(matrix)
    
    N = len(labels)

    # We want to show all ticks...
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    # ... and label them with the respective list entries
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(N):
        for j in range(N):
            text = ax.text(j, i, cm[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Matrice de confusion")
    fig.tight_layout()
    plt.show()
    

show_confusion_matrix(cm, LIST_OF_CLASSES)