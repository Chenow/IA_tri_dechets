# General data parameters
DATASET_PATH = 'dataset-resized'
SHUFFLE_DATA = True

# Data generator parameters
TRAINING_BATCH_SIZE = 16
TRAINING_IMAGE_SIZE = (256, int(1.3*256))
VALIDATION_BATCH_SIZE = 16
VALIDATION_IMAGE_SIZE = (256, int(1.3*256))
TESTING_BATCH_SIZE = 16
TESTING_IMAGE_SIZE = (256, int(1.3*256))
NUMBER_OF_CHANNELS = 3

# Model parameters
RESIDUAL_BLOCKS_PER_MODULE = [3, 3, 1, 0]
FIRST_KERNEL_SIZE = 7
NUMBER_OF_FILTERS_FIRST_BLOCK = 32
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
LIST_OF_CLASSES=["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Target parameters
A_MIN = 0.6
A_C = 0.44
ALPHA = 0.05

#Train parameters
LEARNING_RATE = 1e-4
EPOCHS = 4

#Stockage parameters
PATH_MODELS = "models"
DATA_TO_IDENTIFY_PATH = "datas_to_identify"
PATH_DATAS_IDENTIFIED = "datas_identified"
MODEL_TO_USE = "model1"