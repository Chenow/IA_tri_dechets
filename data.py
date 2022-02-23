import os 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing.image import load_img, smart_resize
from params import *


def create_generators(data_path=DATASET_PATH):
    'Returns three generators'
    image_paths = []
    for folder in os.listdir(data_path):
        paths_to_add = [os.path.join(folder, path) for path in os.listdir(os.path.join(data_path, folder)) if path.endswith('jpg')]
        image_paths = image_paths + paths_to_add

    test_list, val_list, train_list = data_split(np.asarray(image_paths))

    train_data_generator = DataGeneratorClassifier(train_list, TRAINING_BATCH_SIZE, TRAINING_IMAGE_SIZE)
    validation_data_generator = DataGeneratorClassifier(val_list, VALIDATION_BATCH_SIZE, VALIDATION_IMAGE_SIZE)
    test_data_generator = DataGeneratorClassifier(test_list, TESTING_BATCH_SIZE, TESTING_IMAGE_SIZE)
    return train_data_generator, validation_data_generator, test_data_generator


def data_split(paths_list):
    'Splits the paths list into three splits'
    size = TEST_SIZE
    split_1 = size
    split_2 = 2*size
    np.random.seed(seed=SEED)
    np.random.shuffle(paths_list, )
    return paths_list[:split_1], paths_list[split_1:split_2], paths_list[split_2:]


class DataGeneratorClassifier(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size, image_size, data_path=DATASET_PATH, n_channels=NUMBER_OF_CHANNELS, shuffle=SHUFFLE_DATA):
        'Initialisation'
        self.classes = os.listdir(data_path)
        self.image_size = image_size
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.data_path=data_path

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = self.list_IDs[indexes]
        X, y = self.__data_generation(list_IDs_temp)
        
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *image_size, n_channels)
        X = np.empty((self.batch_size, *self.image_size, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):
            
            Xi = load_img(os.path.join(self.data_path, ID))
            Xi = smart_resize(np.asarray(Xi), self.image_size)
            X[i,:] = Xi

            y[i] = self.classes.index(ID.split('/')[0])

        return X,keras.utils.to_categorical(y,num_classes=6)

def show_batch(generator, batch_number=0):
    images, labels = generator.__getitem__(batch_number)
    width = int(np.floor(np.sqrt(labels.shape[0])))
    height = int(np.ceil(labels.shape[0]/float(width)))
    total_height = int(0.09*height*images.shape[1])
    total_width = int(0.09*width*images.shape[2])
    f, axarr = plt.subplots(height,width, figsize=(total_height,total_width))
    for image in range(images.shape[0]):
        image_to_show = (images[image])/np.max(images[image])
        axarr[image//width,image%width].imshow(image_to_show)
        axarr[image//width,image%width].set_title(generator.classes[np.argmax(labels[image])])
    f.tight_layout()
    plt.show()