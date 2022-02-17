from data import create_generators
from model import *
from analyse import *
from sklearn.metrics import confusion_matrix


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
              validation_data=val
                        )


train_model()

cm = confusion_matrix(test[1], model(test[0], normalize='True'))
show_confusion_matrix(cm, LIST_OF_CLASSES)
