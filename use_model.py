from data import *
from params import *
import os
import shutil
import tensorflow as tf
from tensorflow import keras

def identify_images(model_to_use=MODEL_TO_USE):

    model = keras.models.load_model("./" + PATH_MODELS + "/" + MODEL_TO_USE)
    if os.path.exists("./" + PATH_DATAS_IDENTIFIED):
        shutil.rmtree("./" + PATH_DATAS_IDENTIFIED)
    os.makedirs("./" + PATH_DATAS_IDENTIFIED)

    paths_data_to_classify = []
    i = 0
    for img_path in os.listdir("./" + DATA_TO_IDENTIFY_PATH):
        i += 1
        paths_data_to_classify.append("./" + DATA_TO_IDENTIFY_PATH + "/" + img_path)
        img = load_img("./" + DATA_TO_IDENTIFY_PATH + "/" + img_path)
        img_rezised = smart_resize(np.asarray(img), TRAINING_IMAGE_SIZE)
        list_probabilities = model(img_rezised.reshape(-1, *TRAINING_IMAGE_SIZE,NUMBER_OF_CHANNELS))
        class_img = LIST_OF_CLASSES[tf.math.argmax(tf.concat([i for i in list_probabilities],1))]    
        shutil.copyfile("./" + DATA_TO_IDENTIFY_PATH + "/" + img_path, 
                        "./" + PATH_DATAS_IDENTIFIED + "/" + class_img + str(i))

    return
