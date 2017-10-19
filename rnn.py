from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.utils import np_utils
import numpy as np

def preprocess_text(filename, SEQ_LEN):
    text = open(filename).read()
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

    return (char_to_idx, idx_to_char, vocab, text, sentences, next_chars)

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

def build_model(num_layers, seq_len, vocab, layer_size, dropout, lr, decay):
    model = Sequential()
    if(num_layers == 1):
        model.add( LSTM(layer_size, input_shape=(seq_len, vocab), return_sequences=False) )
    else:
        for i in range(num_layers - 1):
            model.add(LSTM(layer_size, input_shape=(seq_len, vocab), return_sequences=True))
        model.add(LSTM(layer_size, return_sequences=False))
    if(dropout):
        model.add(Dropout(dropout))
    model.add(Dense(vocab))
    model.add(Activation('softmax'))
    optimizer = Adam(lr=lr, decay=decay, clipvalue=0.5)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
