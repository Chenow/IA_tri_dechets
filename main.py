from train import *
import os 
from use_model import *

#from utils import test_size
#from params import *


#if __name__ == "__main__":
#  print(test_size(A_MIN, A_C, ALPHA))

import argparse 

parser_mode = argparse.ArgumentParser(description="choisit le mode d'utilisation")
parser_mode.add_argument("mode", metavar="mode", type=str, help="Choisissez le mode d'utilisation, identification ou entrainement ?")

mode = parser_mode.parse_args().mode

if mode == "identification":
  print("Identification des photos...")

  identify_images()
  
  print("Photos identifiées")

elif mode =="entrainement":
  print("Réentraînement du modèle, veuillez patienter...")
  
  train_model()

  print("Modèle réentrainé")

else:
  print("""Error : l'argument "mode" doit être égal à "identification" ou "entrainement" """)

