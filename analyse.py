from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
    
def show_confusion_matrix(labels, test, model):

    x_test, y_test = [], []
    for i in range(len(test)):
      x,y = test[i]
      x = model(x)
      for elm in x:
        x_test.append(elm)
      for elm in y:
        y_test.append(elm)
    cm = confusion_matrix(np.argmax(y_test,axis=1), np.argmax(x_test,axis=1), normalize='true')
    for i in range(6):
      for j in range(6):
        cm[i][j] = round(cm[i][j], 2)

    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm)
    
    N = len(labels)

    # We want to show all ticks...
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    # ... and label them with the respective list entries
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(N):
        for j in range(N):
            text = ax.text(j, i, cm[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Matrice de confusion")
    fig.tight_layout()
    plt.show()
