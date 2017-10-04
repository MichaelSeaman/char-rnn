#!/usr/bin/python

from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, History
from callback import Generate_Text
from sklearn.model_selection import train_test_split
from rnn import preprocess_text, build_model
from frequent_flush import Frequent_flush

import numpy as np
import sys
import os
import getopt
import time

TRAIN_FILE = 'quotes/author-quote.txt'
MODEL_FILE = ''
WEIGHTS_FILE = ''
HISTORY_FILE = ''
NUM_EPOCHS = 30
QUICK_MODE = False
LEARNING_RATE = .001
DECAY = .5
LAYER_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.2
SEQ_LEN = 40
BUFFER_OUTPUT = False

# Dealing with runtime options
def main(argv):
    global TRAIN_FILE
    global MODEL_FILE
    global WEIGHTS_FILE
    global HISTORY_FILE
    global NUM_EPOCHS
    global QUICK_MODE
    global DECAY
    global LEARNING_RATE
    global LAYER_SIZE
    global NUM_LAYERS
    global DROPOUT
    global SEQ_LEN
    global BUFFER_OUTPUT
    try:
        opts, args = getopt.getopt(argv, 'i:e:qm:w:h:b', ['inputFile=','epochs=',
            'quickmode' ,'modelFile=', 'weightsFile=','learningRate=','decay=',
            'numLayers=', 'layerSize=', 'sequenceLength=', 'historyFile=',
            'bufferOutput', 'dropout='])
        for opt, arg in opts:
            if opt in ('-i', '--inputFile'):
                TRAIN_FILE = arg
            elif opt in ('-m', '--modelFile'):
                MODEL_FILE = arg
            elif opt in ('-w', '--weightsFile'):
                WEIGHTS_FILE = arg
            elif opt in ('-h', '--historyFile'):
                HISTORY_FILE = arg
            elif opt in ('-e', '--epochs'):
                NUM_EPOCHS = int(arg)
            elif opt in ('-q', '--quickmode'):
                QUICK_MODE = True
            elif opt == '--learningRate':
                LEARNING_RATE = float(arg)
            elif opt == '--decay':
                DECAY = float(arg)
            elif opt == '--layerSize':
                LAYER_SIZE = int(arg)
            elif opt == '--numLayers':
                NUM_LAYERS = int(arg)
            elif opt == '--dropout':
                DROPOUT = float(arg)
            elif opt == '--sequenceLength':
                SEQ_LEN = int(arg)
            elif opt in ('-b', '--bufferOutput'):
                BUFFER_OUTPUT = True
    except getopt.GetoptError as e:
        print(e)

    if(not BUFFER_OUTPUT):
        Frequent_flush(1).start()

    print("Using Input File: ", TRAIN_FILE)
    print("Preprocessing...")
    X, y, char_to_idx, idx_to_char, vocab, text = preprocess_text(
        filename=TRAIN_FILE, SEQ_LEN=SEQ_LEN)

    if(QUICK_MODE):
        X = X[:1000]
        y = y[:1000]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Building a simple LSTM
    if(MODEL_FILE):
        print("Loading model")
        model = load_model(MODEL_FILE)
    else:
        print("Building model")
        print("Size of Layers: ", LAYER_SIZE)
        print("Depth of Network: ", NUM_LAYERS)
        print("Using dropout: ", DROPOUT)

        model = build_model(NUM_LAYERS=NUM_LAYERS, SEQ_LEN=SEQ_LEN, vocab=vocab,
            LAYER_SIZE=LAYER_SIZE, DROPOUT=DROPOUT)

    if(WEIGHTS_FILE):
        model.load_weights(WEIGHTS_FILE)
    optimizer = Adam(lr=LEARNING_RATE, decay=DECAY, clipvalue=0.5)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.summary()

    if(HISTORY_FILE):
        print("Using history file ", HISTORY_FILE)
        history = np.load(HISTORY_FILE).item()
    else:
        history = {'acc':[], 'val_acc':[], 'loss':[], 'val_loss':[]}


    # Setting up models directory
    if not os.path.exists('models'):
        os.makedirs('models')

    timestr = time.strftime("%m%d-%H%M")
    model_name = "CharRNN_" + timestr
    model_dir = os.path.join('models', model_name)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print("Saving model data to:", os.path.abspath(model_dir))
        model.save(os.path.join(model_dir, "model.h5"))

    print("Creating callbacks")



    weights_filepath = os.path.join(model_dir,
        "weights-improvement-{epoch:03d}-{loss:.4f}.hdf5")
    checkpoint = ModelCheckpoint(weights_filepath, save_best_only=True,
        verbose=1)
    generate_text = Generate_Text(gen_text_len=300, idx_to_char=idx_to_char,
        char_to_idx=char_to_idx, text=text, maxlen=SEQ_LEN, vocab=vocab)
    esCallback = EarlyStopping(min_delta=0, patience=10, verbose=1)
    hisCallback = History()
    if(QUICK_MODE):
        callbacks_list = [checkpoint, hisCallback]
    else:
        callbacks_list = [checkpoint, generate_text, hisCallback]


    print("Training model")
    for e in range(NUM_EPOCHS):
        try:
            print("\nEPOCH {}\n".format(e))
            hist = model.fit(X_train, y_train, validation_data=(X_test,y_test),
                batch_size=128, epochs=1, callbacks=callbacks_list)
            for k, v in hist.history.items():
                history[k] = history[k] + v
        except KeyboardInterrupt:
            print("Exiting training loop")
            break
    print("Saving training history")
    history_filename = os.path.join(model_dir, 'model_history.npy')
    np.save(history_filename, history)


if __name__ == "__main__":
    main(sys.argv[1:])
