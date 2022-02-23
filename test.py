from params import *
from analyse import show_confusion_matrix
from utils import interval_confiance
from tensorflow import keras
from train import get_data


def test_model():
  model = keras.models.load_model("./" + PATH_MODELS + "/" + MODEL_TO_USE)
  test = get_data((*TRAINING_IMAGE_SIZE,NUMBER_OF_CHANNELS))[2]
  moyenne = model.evaluate(test, return_dict=True)["accuracy"]

  show_confusion_matrix(test, model)

  interval_confiance(moyenne, TEST_SIZE, ALPHA)
  return

