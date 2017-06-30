import numpy as np
import sys
from keras.callbacks import Callback

class Generate_Text(Callback):

    def __init__(self, gen_text_len, idx_to_char, char_to_idx, text, maxlen, vocab):
        self.gen_text_len = gen_text_len
        self.idx_to_char = idx_to_char
        self.char_to_idx = char_to_idx
        self.text = text
        self.maxlen = maxlen
        self.vocab = vocab

    def on_epoch_end(self, epoch, logs={}):

        start_index = np.random.randint(0, len(self.text) - self.maxlen - 1)

        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('----- diversity:', diversity)
            generated = ''
            sentence = self.text[start_index: start_index + self.maxlen]
            print('----- Generating with seed: "' + sentence + '"')

            for i in range(self.gen_text_len):
                x = np.zeros((1, self. maxlen, self.vocab))
                for t, char in enumerate(sentence):
                    x[0,t,self.char_to_idx[char]] = 1.

                preds = self.model.predict(x,verbose=0)[0]
                next_idx = sample(preds, diversity)
                next_char = self.idx_to_char[next_idx]
                generated += next_char
                sentence = sentence[1:] + next_char
                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1)
    return np.argmax(probs)
