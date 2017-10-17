#!/usr/bin/env python

from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, History
from callback import *
from sklearn.model_selection import train_test_split
from rnn import preprocess_text, build_model, vectorize_data
from frequent_flush import Frequent_flush
from model_file import ModelFile

import numpy as np
import sys
import os
import getopt
import time

TRAIN_FILE = 'quotes/author-quote.txt'
MODEL_DIR = ''
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

    training_file, new_model, model_dir, num_epochs, quick_mode, \
        learning_rate, decay, layer_size, num_layers, dropout, seq_len, \
        buffer_output = parse_options(argv)

    if(not buffer_output):
        Frequent_flush(1).start()

    print("Using Input File: ", training_file)
    print("Preprocessing...")
    char_to_idx, idx_to_char, vocab, text, sentences, next_chars = \
        preprocess_text(filename=training_file, SEQ_LEN=SEQ_LEN)
    X, y = vectorize_data(SEQ_LEN, vocab, sentences, char_to_idx, next_chars)

    if(quick_mode):
        X = X[:1000]
        y = y[:1000]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Building a simple LSTM
    if new_model:
        print("Building model")
        print("Size of Layers: ", layer_size)
        print("Depth of Network: ", num_layers)
        print("Using dropout: ", dropout)

        model = build_model(num_layers=num_layers, seq_len=seq_len, vocab=vocab,
            layer_size=layer_size, dropout=dropout, lr=LEARNING_RATE,
            decay=DECAY)

        model.summary()

        model_dir = create_model_dir(model)
        print("Saved model data to:", os.path.abspath(model_dir))

        history = {'acc':[], 'val_acc':[], 'loss':[], 'val_loss':[]}
        initial_epoch = 0


    else:
        model, history, initial_epoch = load_model_dir(model_dir)
        model.summary()

    print("Creating callbacks")
    callbacks_list = create_callbacks(model_dir, writer=True,
        idx_to_char=idx_to_char, char_to_idx=char_to_idx, text=text,
        seq_len=SEQ_LEN, vocab=vocab)

    print("\nTraining model\n")
    history = train_model(model, X_train, y_train, X_test, y_test,
        callbacks_list, num_epochs, history, initial_epoch)

    print("Saving training history")
    history_filename = os.path.join(model_dir, 'model_history.npy')
    np.save(history_filename, history)

def create_model_dir(model):
    if not os.path.exists('models'):
        os.makedirs('models')

    timestr = time.strftime("%m%d-%H%M")
    model_name = "CharRNN_" + timestr
    model_dir = os.path.join('models', model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save(os.path.join(model_dir, "model.h5"))
    return model_dir

def load_model_dir(model_dir):
    model_file = ModelFile(model_directory=model_dir)
    if(model_file.check_exists()):
        print("Loading model")
    else:
        print("Incomplete model. Make sure that the folder exists and" +
            " contains weights, history, and model.h5")
        sys.exit(0)

    model_dir = model_file.model_directory
    model_filename = model_file.model
    weights_file = model_file.weights
    history_file = model_file.history

    model = load_model(model_filename)
    print("Using weights file ", weights_file)
    model.load_weights(weights_file)
    print("Using history file ", history_file)
    history = np.load(history_file).item()
    initial_epoch = len(history['acc'])
    print("Starting from epoch ", initial_epoch)
    return model,history,initial_epoch

def create_callbacks(model_dir, history=True, checkp=True, earlyStop=False,
        writer=False, idx_to_char = {}, char_to_idx = {}, text = "",
        seq_len = 300 , vocab = 0):
    callbacks = []
    weights_filepath = os.path.join(model_dir,
        "weights-improvement-{epoch:03d}-{loss:.4f}.hdf5")

    composition_dir = os.path.join(model_dir, 'compositions')
    composition_filepath = os.path.join(composition_dir,
        "epoch_{epoch:03d}_composition_reduced.txt")

    checkpoint = ModelCheckpoint(weights_filepath, save_best_only=True,
        verbose=1)
    esCallback = EarlyStopping(min_delta=0, patience=10, verbose=1)
    hisCallback = History()
    writerCallback = Write_Text(idx_to_char, char_to_idx, text, seq_len, vocab,
            filepath=composition_filepath)
    if history:
        callbacks.append(hisCallback)
    if checkp:
        callbacks.append(checkpoint)
    if earlyStop:
        callbacks.append(esCallback)
    if writer:
        if not os.path.exists(composition_dir):
            os.makedirs(composition_dir)
        callbacks.append(writerCallback)

    return callbacks

def train_model(model, X_train, y_train, X_test, y_test, callbacks_list,
 num_epochs, history, initial_epoch=0):
    for e in range(num_epochs):
        epochs = e + initial_epoch
        try:
            print("\nEPOCH {}\n".format(epochs))
            hist = model.fit(X_train, y_train, validation_data=(X_test,y_test),
                batch_size=128, epochs=epochs+1, callbacks=callbacks_list,
                initial_epoch=epochs)
            for k, v in hist.history.items():
                history[k] = history[k] + v
        except KeyboardInterrupt:
            print("Exiting training loop")
            break
    return history

def parse_options(argv):
    '''
    Takes a list of strings, and returns a list of hyper parameters in the order
    given below
    '''
    global TRAIN_FILE
    global MODEL_DIR
    global NUM_EPOCHS
    global QUICK_MODE
    global DECAY
    global LEARNING_RATE
    global LAYER_SIZE
    global NUM_LAYERS
    global DROPOUT
    global SEQ_LEN
    global BUFFER_OUTPUT
    new_model = True

    try:
        opts, args = getopt.getopt(argv, 'i:e:qm:w:h:b', ['inputFile=','epochs=',
            'quickmode' , 'learningRate=','decay=', 'numLayers=', 'layerSize=',
            'sequenceLength=', 'historyFile=', 'bufferOutput', 'dropout=',
            'modelDirectory'])
        for opt, arg in opts:
            if opt in ('-i', '--inputFile'):
                TRAIN_FILE = arg
            elif opt in ('-m', '--modelDirectory'):
                MODEL_DIR = arg
                new_model = False
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
    return (TRAIN_FILE, new_model, MODEL_DIR, NUM_EPOCHS, QUICK_MODE,
                LEARNING_RATE, DECAY, LAYER_SIZE, NUM_LAYERS, DROPOUT,
                SEQ_LEN, BUFFER_OUTPUT)



if __name__ == "__main__":
    main(sys.argv[1:])
