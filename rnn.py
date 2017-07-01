'''
Copied from keras tutorials' char-rnn


Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, History
from callback import Generate_Text
from sklearn.model_selection import train_test_split

import numpy as np
import random
import sys
import os
import getopt
import time

TRAIN_FILE = '/home/ubuntu/Datasets/quote.txt'
WEIGHTS_FILE = ''
NUM_EPOCHS = 30
QUICK_MODE = False

def main(argv):
    global TRAIN_FILE
    global WEIGHTS_FILE
    global NUM_EPOCHS
    global CONTINUE_FROM_PREV
    try:
        opts, args = getopt.getopt(argv, 't:e:qw:', ['trainFile=','epochs=','quickmode' ,'weightsFile='])
        for opt, arg in opts:
            if opt in ('-t', '--trainFile'):
                TRAIN_FILE = arg
            elif opt in ('-w', '--weightsFile'):
                WEIGHTS_FILE = arg
            elif opt in ('-e', '--epochs'):
                NUM_EPOCHS = int(arg)
            elif opt in ('-q', '--quickmode'):
                QUICK_MODE = True
    except getopt.GetoptError as e:
        print("No train/weights file provided")
        print(e)

    print("Using Training File: ", TRAIN_FILE)
    assert(os.path.exists(TRAIN_FILE))

    if(WEIGHTS_FILE):
        print("Continuing from previous training.")
        print("Using Weights File: ", WEIGHTS_FILE)
        assert(os.path.exists(WEIGHTS_FILE))

    print("Training epochs: ", NUM_EPOCHS)

    if(QUICK_MODE):
        print("Quick mode enabled.")

    text = open(TRAIN_FILE).read().lower()
    n_chars = len(text)
    print('Corpus length: ', n_chars)

    chars = sorted(list(set(text)))
    vocab = len(chars)
    print('Total unique chars: ', vocab)

    char_to_idx = dict( (c, i) for i,c in enumerate(chars) )
    idx_to_char = dict( (i, c) for i,c in enumerate(chars) )

    maxlen = 40
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, n_chars - maxlen, step):
        sentences.append(text[i:i+maxlen])
        next_chars.append(text[i+maxlen])

    n_data = len(sentences)
    print("Number of sequences: ", n_data)

    print("Vectorizing")
    X = np.zeros( (n_data, maxlen, vocab), dtype=np.bool)
    y = np.zeros( (n_data, vocab), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i,t, char_to_idx[char]] = 1
        y[i, char_to_idx[ next_chars[i] ]] = 1

    if(QUICK_MODE):
        X = X[:1000]
        y = y[:1000]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    # Building a simple LSTM
    print("Building model")
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, vocab), return_sequences=True ))
    model.add(LSTM(128, input_shape=(maxlen, vocab)))
    model.add(Dropout(0.2))
    model.add(Dense(vocab))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.1)
    if(WEIGHTS_FILE):
        model.load_weights(WEIGHTS_FILE)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    print("Creating callbacks")

    filepath = "weights/weights-improvement-{epoch:03d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, save_best_only=True)
    generate_text = Generate_Text(gen_text_len=100, idx_to_char=idx_to_char, char_to_idx=char_to_idx,text=text, maxlen=maxlen, vocab=vocab)
    esCallback = EarlyStopping(min_delta=0, patience=10, verbose=1)
    hisCallback = History()
    if(QUICK_MODE):
        callbacks_list = [checkpoint, hisCallback]
    else:
        callbacks_list = [checkpoint, generate_text, hisCallback]

    if not os.path.exists('weights'):
        os.makedirs('weights')

    print("Training model")
    hist = model.fit(X_train, y_train, validation_data=(X_test,y_test), batch_size=128, epochs=NUM_EPOCHS, callbacks=callbacks_list)
    timestr = time.strftime("%m%d-%H%M")
    np.save('model_history_' + timestr + '.npy', hist.history)


if __name__ == "__main__":
    main(sys.argv[1:])
