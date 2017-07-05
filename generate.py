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
from keras.models import model_from_json
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM


import numpy as np
import random
import sys
import os
import getopt

TRAIN_FILE = '/home/ubuntu/Datasets/quote.txt'
WEIGHTS_FILE = './weights/'
MODEL_FILE = 'model.json'
diversity = 1.0

def main(argv):
    global TRAIN_FILE
    global WEIGHTS_FILE
    global diversity
    global MODEL_FILE
    try:
        opts, args = getopt.getopt(argv, 't:w:d:m:', ['trainFile=','weightsFile=','diversity=','modelFile='])
        for opt, arg in opts:
            if opt in ('-t', '--trainFile'):
                TRAIN_FILE = arg
            elif opt in ('-w', '--weightsFile'):
                WEIGHTS_FILE = arg
            elif opt in ('-d', '--diversity'):
                diversity = float(arg)
            elif opt in ('-m', '--modelFile'):
                MODEL_FILE = arg
    except getopt.GetoptError as e:
        print("No train/weights file provided")
        print(e)

    print("Using Training File: ", TRAIN_FILE)
    assert(os.path.exists(TRAIN_FILE))

    print("Using Weights File: ", WEIGHTS_FILE)
    assert(os.path.exists(WEIGHTS_FILE))

    print("Using Model File: ", MODEL_FILE)
    assert(os.path.exists(MODEL_FILE))

    text = open(TRAIN_FILE).read().lower()
    n_chars = len(text)
    print('Corpus length: ', n_chars)

    chars = sorted(list(set(text)))
    vocab = len(chars)
    print('Total unique chars: ', vocab)

    char_to_idx = dict( (c, i) for i,c in enumerate(chars) )
    idx_to_char = dict( (i, c) for i,c in enumerate(chars) )

    print("Building model")
    with open(MODEL_FILE) as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(WEIGHTS_FILE)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    maxlen = model.layers[0].batch_input_shape[1]

    gen_text_len = int(input('Length of generated text: '))
    output_file = input('Enter output filename (Leave blank to leave text unsaved) : ')
    output_file_desired = output_file != ''
    input_seed = input('Enter an input seed, max 40 chars: ').lower()

    sentence = " " * maxlen + input_seed
    sentence = sentence[-1*maxlen:]

    print("Seed: '", sentence, "'")
    generated = ""
    for i in range(gen_text_len):
        x = np.zeros((1, maxlen, vocab))
        for t, char in enumerate(sentence):
            x[0,t,char_to_idx[char]] = 1.

        preds = model.predict(x,verbose=0)[0]
        next_idx = sample(preds, diversity)
        next_char = idx_to_char[next_idx]
        generated += next_char
        sentence = sentence[1:] + next_char
    print(generated)
    if(output_file_desired):
        with open(output_file) as f:
            f.write(generated)

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1)
    return np.argmax(probs)

if __name__ == "__main__":
    main(sys.argv[1:])
