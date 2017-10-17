import numpy as np
import matplotlib.pyplot as plt
import os
import getopt
import sys

HISTORY_FILE = 'model_history.npy'

def main(argv):
    global HISTORY_FILE
    if(argv):
        inputFile = argv[0]
    else:
        print("History file or model directory needed")
        sys.exit(0)

    if(os.path.isdir(inputFile)):
        model_name = os.path.basename(inputFile)
        HISTORY_FILE = os.path.join(inputFile, HISTORY_FILE)
    else:
        model_name = 'model'
        HISTORY_FILE = inputFile

    print("Model: ", model_name)
    print("Using History File: ", HISTORY_FILE)


    history = np.load(HISTORY_FILE).item()
    plot_history(history, model_name)

    print("Saved output to ", os.path.abspath(model_name + '.png'))

def plot_history(history, model_name):
    # summarize history for accuracy
    plt.figure(1)
    plt.subplot(211)
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('{model_name} accuracy'.format(model_name=model_name))
    plt.ylabel('accuracy')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    plt.subplot(212)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('{model_name} loss'.format(model_name=model_name))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.tight_layout()
    plt.savefig('{model_name}.png'.format(model_name=model_name))

if __name__ == "__main__":
    main(sys.argv[1:])
