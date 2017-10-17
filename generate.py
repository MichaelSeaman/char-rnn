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
from keras.models import load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from model_file import ModelFile
from rnn import preprocess_text
from keras.utils import np_utils


import numpy as np
import random
import sys
import os
import getopt

TRAIN_FILE = 'quotes/author-quote.txt'
MODEL_DIR = ''
diversity = 1.0

def main(argv):
    global TRAIN_FILE
    global diversity
    global MODEL_DIR
    try:
        opts, args = getopt.getopt(argv, 't:d:m:', ['trainFile=','diversity=',
            'modelDirectory='])
        for opt, arg in opts:
            if opt in ('-t', '--trainFile'):
                TRAIN_FILE = arg
            elif opt in ('-m', '--modelDirectory'):
                MODEL_DIR = arg
            elif opt in ('-d', '--diversity'):
                diversity = float(arg)
    except getopt.GetoptError as e:
        print("No train/weights file provided")
        print(e)

    print("Using Training File: ", TRAIN_FILE)
    assert(os.path.exists(TRAIN_FILE))

    print("Using model: ", MODEL_DIR)
    model_file = ModelFile(model_directory=MODEL_DIR)
    if(model_file.check_exists()):
        print("Loading model")
    else:
        print("Incomplete model. Make sure that the folder exists and" +
            " contains weights, and model.h5")
        sys.exit(0)

    model_dir = model_file.model_directory
    model_filename = model_file.model
    weights_file = model_file.weights

    print("Building model")
    model = load_model(model_filename)
    model.load_weights(weights_file)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    maxlen = model.layers[0].batch_input_shape[1]

    char_to_idx, idx_to_char, vocab, text, sentences, next_chars = \
        preprocess_text(filename=TRAIN_FILE, SEQ_LEN= maxlen)

    gen_text_len = int(input('Length of generated text: '))
    output_file = input('Enter output filename (Leave blank to leave text unsaved) : ')
    output_file_desired = output_file != ''
    input_seed = input('Enter an input seed, max 40 chars: ').lower()

    sentence = "a" * maxlen + input_seed
    sentence = sentence[-1*maxlen:]

    print("Seed: '", sentence, "'")
    generated = generate(model, idx_to_char, char_to_idx, gen_text_len, maxlen,
        vocab, sentence)
    print(generated)
    if(output_file_desired):
        with open(output_file) as f:
            f.write(generated)



def generate(model, idx_to_char, char_to_idx, gen_text_len, seq_len, vocab,
    sentence="", text="", temperature=.5):
    generated = ""
    vectorized_char_to_idx = np.vectorize(char_to_idx.get)

    if not sentence:
        start_index = np.random.randint(0, len(text) - seq_len - 1)
        sentence = text[start_index: start_index + seq_len]

    for i in range(gen_text_len):
        sentence_nested = [list(sentence)]
        X = vectorized_char_to_idx(sentence_nested)
        X = np_utils.to_categorical(X, num_classes=vocab)
        X = X.reshape((1, seq_len, vocab))

        preds = model.predict(X,verbose=0)[0]
        next_idx = sample(preds, temperature=temperature)
        next_char = idx_to_char[next_idx]
        generated += next_char
        sentence = sentence[1:] + next_char

    return generated

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
