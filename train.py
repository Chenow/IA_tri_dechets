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

