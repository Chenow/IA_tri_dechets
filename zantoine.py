import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.datasets import cifar10
from keras.optimizers import Adam


# Model parameters
PADDING_MODEL = "same"
ACTIVATION_MODEL = "relu"
KERNEL_SIZE_MODEL = (3, 3)
POOL_SIZE_MODEL = (2, 2)
NBR_CONV_MODEL = [3, 3, 5, 2]

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape(-1, 32, 32, 3).astype("float32") / 255.0
x_test = x_test.reshape(-1, 32, 32, 3).astype("float32") / 255.0



class block_model(layers.Layer):

  def __init__(self):
    super(block_model, self).__init__()
    self.MaxPooling2D = keras.layers.MaxPooling2D(pool_size=POOL_SIZE_MODEL, padding=PADDING_MODEL)

  def call(self, x, nbr_conv, nbr_filtres):
    x=self.MaxPooling2D(x)
    for i in range(2):
      x =  keras.layers.Conv2D(nbr_filtres, kernel_size=KERNEL_SIZE_MODEL, padding=PADDING_MODEL, activation=ACTIVATION_MODEL)(x)
    for i in range(nbr_conv):
      y =  keras.layers.Conv2D(nbr_filtres, kernel_size=KERNEL_SIZE_MODEL, padding=PADDING_MODEL, activation=ACTIVATION_MODEL)(x)
      x +=  keras.layers.Conv2D(nbr_filtres, kernel_size=KERNEL_SIZE_MODEL, padding=PADDING_MODEL, activation=ACTIVATION_MODEL)(y)

    return x
class ResNet(keras.Model):
 
  def __init__(self, 
               padding=PADDING_MODEL,
               activation=ACTIVATION_MODEL,
               pool_size=POOL_SIZE_MODEL):
    super(ResNet, self).__init__()
    self.conv2D_unique = keras.layers.Conv2D(64, 7, 7, padding=padding, activation=activation)
    self.AveragePooling2D = keras.layers.AveragePooling2D(pool_size=pool_size, padding=padding)
    self.block_model = block_model()

  def call(self, 
          input_tensor,
          nbr_conv=NBR_CONV_MODEL):

    x = self.conv2D_unique(input_tensor)
    for i in range(len(nbr_conv)):
      x = self.block_model(x,
                      nbr_conv[i],
                      64*2**i,
                      )
    x = self.AveragePooling2D(x)
    return self




model = ResNet()
model.summary()
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=32, epochs=2, verbose=2)
model.evaluate(x_test, y_test, batch_size=32)

