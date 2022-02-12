import tensorflow as tf
from tensorflow import keras
import numpy as np
from params import *


#ResNet

def residual_block(x_input, filter_numbers, kernel_size=3, activation='relu'):
    x_intermediate = tf.keras.layers.Conv2D(filter_numbers, kernel_size, padding="same", activation=activation)(x_input)
    return tf.keras.layers.Conv2D(filter_numbers, kernel_size, padding="same", activation=activation)(x_intermediate) + x_input


def module(x, number_residual_blocks, filter_numbers,
    number_filters_first_block, kernel_size=3, activation='relu'):
    x = tf.keras.layers.MaxPooling2D(padding="same", pool_size=(2, 2))(x)

    if filter_numbers != number_filters_first_block:
        x = tf.keras.layers.Conv2D(filter_numbers,
         kernel_size, padding="same", activation=activation)(x)
        x = tf.keras.layers.Conv2D(filter_numbers, kernel_size, padding="same",
         activation=activation)(x)

    for i in range(number_residual_blocks):
        x = residual_block(x, filter_numbers=filter_numbers)

    return x


def calculate_probability(x):
    x = tf.keras.layers.Conv2D(NUMBER_OF_FILTERS_FIRST_BLOCK, FIRST_KERNEL_SIZE, 
    activation=ACTIVATION)(x)

    for index, number_of_residual_blocks in enumerate(RESIDUAL_BLOCKS_PER_MODULE):
        x = module(x,
                   number_residual_blocks=number_of_residual_blocks,
                   filter_numbers=NUMBER_OF_FILTERS_FIRST_BLOCK*2**index,
                   number_filters_first_block=NUMBER_OF_FILTERS_FIRST_BLOCK,
                   kernel_size=KERNEL_SIZE,
                   activation=ACTIVATION) 

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    return tf.keras.layers.Dense(NUMBER_OF_CLASSES, activation="softmax")(x)



#MobileNet

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

