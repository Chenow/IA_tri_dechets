from data import create_generators
from model import *


def get_data(input_shape):
    train, val, test = create_generators()
    return train, val, test

def get_model(input_shape):
    x_input = tf.keras.layers.Input(input_shape)
    x_output = calculate_probability_mobile_netv2(x_input)
    return tf.keras.Model(inputs=x_input, outputs=x_output)

def train_model():
    train, val, test = get_data(INPUT_SHAPE)
    model = get_model(INPUT_SHAPE)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=["accuracy"])
    model.fit(train, epochs=5, validation_data=test)
 
train_model()