# Use

This project allows you to train and identify a CNN following the ResNet model.

### Training a model

- put your dataset in "dataset-rezised", and class the datas by creating a subfolder per category.
- Enter the corrrect values for the variables NUMBER_OF_CLASSES and LIST_OF_CLASSES in the file "params.py".
- Run ``` 'python3 main.py entrainement' ```.
- Your model is now saved in the folder "models" as the one with the higher number.

### Identify datas
- put the datas to be identified inside the folder "datas_to_identify".
- Run ``` 'python3 main.py identification' ```.
- Your identified datas are now saved in "datas_identified".
