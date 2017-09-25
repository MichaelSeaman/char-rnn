from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.utils import np_utils
import numpy as np

def preprocess_text(filename, SEQ_LEN):
    text = open(filename).read().lower()
    n_chars = len(text)
    chars = sorted(list(set(text)))
    vocab = len(chars)

    char_to_idx = dict( (c, i) for i,c in enumerate(chars) )
    idx_to_char = dict( (i, c) for i,c in enumerate(chars) )

    step = 3
    sentences = []
    next_chars = []
    for i in range(0, n_chars - SEQ_LEN, step):
        sentences.append("".join(text[i:i+SEQ_LEN]))
        next_chars.append(text[i+SEQ_LEN])

    print("Vectorizing")
    X, y = vectorize_data(SEQ_LEN, vocab, sentences, char_to_idx, next_chars)
    return X, y, char_to_idx, idx_to_char, vocab, text

def vectorize_data(SEQ_LEN, vocab, sentences, char_to_idx, next_chars):
    n_data = len(sentences)
    vectorized_char_to_idx = np.vectorize(char_to_idx.get)
    sentences_np = np.asarray([list(sentence) for sentence in sentences])
    X = vectorized_char_to_idx(sentences_np)
    X = np_utils.to_categorical(X, num_classes=vocab)
    X = X.reshape( (n_data, SEQ_LEN, vocab) )

    y = np_utils.to_categorical(vectorized_char_to_idx(
        np.asarray(next_chars)), num_classes=vocab)

    return X, y

def build_model(NUM_LAYERS, SEQ_LEN, vocab, LAYER_SIZE, DROPOUT):
    model = Sequential()
    if(NUM_LAYERS == 1):
        model.add( LSTM(LAYER_SIZE, input_shape=(SEQ_LEN, vocab), return_sequences=False) )
    else:
        for i in range(NUM_LAYERS - 1):
            model.add(LSTM(LAYER_SIZE, input_shape=(SEQ_LEN, vocab), return_sequences=True))
        model.add(LSTM(LAYER_SIZE, return_sequences=False))
    if(DROPOUT):
        model.add(Dropout(DROPOUT))
    model.add(Dense(vocab))
    model.add(Activation('softmax'))
    return model
