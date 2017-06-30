import numpy as np
import matplotlib.pyplot as plt
import os
import getopt
import sys

HISTORY_FILE = 'model_history_0630-1530.npy'


def main(argv):
    global HISTORY_FILE
    try:
        opts, args = getopt.getopt(argv, 'h:', ['historyFile='])
        for opt, arg in opts:
            if opt in ('-h', '--historyFile'):
                HISTORY_FILE = arg
    except getopt.GetoptError as e:
        print("No train/weights file provided")
        print(e)

    print("Using History File: ", HISTORY_FILE)
    assert(os.path.exists(HISTORY_FILE))

    h = np.load(HISTORY_FILE).item()

    print(h.keys())
    # summarize history for accuracy
    plt.plot(h['acc'])
    plt.plot(h['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('model_acc.png')
    # summarize history for loss
    plt.close()
    plt.plot(h['loss'])
    plt.plot(h['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('model_loss.png')


if __name__ == "__main__":
    main(sys.argv[1:])
