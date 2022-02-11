from numpy import block
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.datasets import cifar10
from keras import optimizers
from keras import losses

# Model parameters
PADDING_MODEL = "same"
ACTIVATION_MODEL = "relu"
KERNEL_SIZE_MODEL = (3, 3)
POOL_SIZE_MODEL = (2, 2)
NBR_CONV_MODEL = [3, 4, 5, 2]

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape(-1, 32, 32, 3).astype("float32") / 255.0
x_test = x_test.reshape(-1, 32, 32, 3).astype("float32") / 255.0



class block(layers.layer)




class block_ResNet(layers.Layer):
    def __init__(self, nbr_filtres, pool_size=POOL_SIZE_MODEL, padding=PADDING_MODEL,
     kernel_size=KERNEL_SIZE_MODEL, activation=ACTIVATION_MODEL):
        super(block_ResNet, self).__init__()
        self.MaxPooling2D = layers.MaxPooling2D(pool_size=pool_size, padding=padding)

        self.Conv2D = layers.Conv2D(nbr_filtres, kernel_size=kernel_size, 
        padding=padding, activation=activation)

    def call(self, input_tensor, nbr_conv):
        x = self.MaxPooling2D(input_tensor)
        for i in range(2):
            x = self.Conv2D(x)
        for i in range(nbr_conv):
            x = self.lacouche()
            x+= self.Conv2D(y)
        return x



class ResNet(keras.Model):
    def __init__(self, num_classes=6, padding=PADDING_MODEL, activation=ACTIVATION_MODEL, pool_size=POOL_SIZE_MODEL):
        super(ResNet, self).__init__()
        self.Conv2D_unique = layers.Conv2D(64, 7, 7, padding=padding, activation=activation)
        self.AveragePooling2D = keras.layers.AveragePooling2D(pool_size=pool_size,padding=padding)
        self.block_Resnet1 = block_ResNet(64)
        self.block_Resnet2 = block_ResNet(128)
        self.block_Resnet3 = block_ResNet(256)
        self.block_Resnet4 = block_ResNet(64)

    def call(self, input_tensor, nbr_conv=NBR_CONV_MODEL):
        x = self.Conv2D_unique(input_tensor)
        x =  self.block_Resnet1(x, nbr_conv[0])
        x =  self.block_Resnet2(x, nbr_conv[1])
        x =  self.block_Resnet3(x, nbr_conv[2])
        x =  self.block_Resnet4(x, nbr_conv[3])
        return self.AveragePooling2D(x)


model=ResNet()
model.build((1, 32, 32, 3))
model.summary()
model.compile(optimizer=keras.optimizers.Adam(),
 loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
model.fit(x_train, y_train)