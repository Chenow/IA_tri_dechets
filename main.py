import argparse 
import os 

from train import train_model
from infer import identify_images
from test import test_model


if __name__ == "__main__":

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

  elif mode =="test":
    test_model()
  else:
    print("""Error : l'argument "mode" doit être égal à "identification", "entrainement", ou "test" """)

