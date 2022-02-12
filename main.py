import tensorflow as tf
from tensorflow import keras
import numpy as np
from data import create_generators

# Model parameters
RESIDUAL_BLOCKS_PER_MODULE = [3, 3, 5, 2]
FIRST_KERNEL_SIZE = 7
NUMBER_OF_FILTERS_FIRST_BLOCK = 64
ACTIVATION = "relu"
KERNEL_SIZE = 3
FIRST_CHANNEL = 32
FIRST_STRIDE = 2
EXPANSION_FACTORS = [1, 6, 6, 6, 6, 6, 6]
CHANNELS = [16, 24, 32, 64, 96, 160, 320]
ITERATIONS = [1, 2, 3, 4, 3, 3, 1]
STRIDES = [1, 2, 2, 2, 1, 2, 1]
LAST_CHANNEL = 1280

# Data parameters
NUMBER_OF_CLASSES = 6
INPUT_SHAPE = (32, 32, 3)


def bottleneck(x, t, c, n, s, activation="relu", kernel_size=3):
    for i in range(n):
        x0 = x
        x = tf.keras.layers.Conv2D(c*t, 1, activation=activation)(x)
        x = tf.keras.layers.DepthwiseConv2D(kernel_size, padding='same',
                                            activation=activation,
                                            strides=1 if i != 0 else s)(x)
        x = tf.keras.layers.Conv2D(c, 1)(x)   
        if i != 0:
            x += x0
    return x


def calculate_probability_mobile_netv2(x0, activation=ACTIVATION, kernel_size=KERNEL_SIZE):
    x = tf.keras.layers.Conv2D(FIRST_CHANNEL, kernel_size, padding="same", activation=activation,
                               strides=FIRST_STRIDE)(x0)

    for t, c, n, s in zip(EXPANSION_FACTORS,
                          CHANNELS,
                          ITERATIONS,
                          STRIDES):
        x = bottleneck(x, t, c, n, s, activation=activation, kernel_size=kernel_size)

    x = tf.keras.layers.Conv2D(LAST_CHANNEL, kernel_size, padding="same", activation=activation)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(NUMBER_OF_CLASSES, activation='softmax')(x)
    return x







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
                  loss=tf.keras.losses.CategoricalCrossentropy())
    model.fit(train)
 
train_model()
