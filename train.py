from bdb import effective
from data import create_generators
from params import *
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
import os


def get_data(input_shape):
    train, val, test = create_generators()
    return train, val, test

def get_model(input_shape):
    x_input = tf.keras.layers.Input(input_shape)
    x_output = calculate_probability(x_input)
    return tf.keras.Model(inputs=x_input, outputs=x_output)

def save_model(model, path_models = PATH_MODELS):
    if os.listdir("./" + path_models) == []:
           model.save("./" + path_models + "/model1")
    else:
        name_last_model = os.listdir("./" + path_models)[-1]
        path_save_model = "./" + path_models + "/model" + str(int(name_last_model[-1]) + 1) 
        model.save(path_save_model)
    return

def train_model(epochs=EPOCHS, learning_rate=LEARNING_RATE):
    train, val, test = get_data((*TRAINING_IMAGE_SIZE,NUMBER_OF_CHANNELS))
    model = get_model((*TRAINING_IMAGE_SIZE,NUMBER_OF_CHANNELS))
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=["accuracy"])
    model.fit(train,
              epochs=epochs,
              steps_per_epoch=1,
              validation_data=val
                        )
    save_model(model)
    return test, model

