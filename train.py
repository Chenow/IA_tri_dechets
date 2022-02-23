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

    if not os.path.exists("./" + path_models):
        os.makedirs("./" + path_models)

    if not os.listdir("./" + path_models):
           model.save("./" + path_models + "/model_1")

    else:
        print(os.listdir(path_models))
        indice_last_model = max([int(i.split("_")[-1]) for i in os.listdir(path_models) if not i.startswith(".") ]) 
        path_save_model = "./" + path_models + "/model_" + str(indice_last_model + 1) 
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
              steps_per_epoch=94,
              validation_data=val
                        )
    model.evaluate(test)
    save_model(model)
    return test, model

