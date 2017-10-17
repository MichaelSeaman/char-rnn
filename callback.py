import numpy as np
from keras.utils import np_utils
import sys
import os
from keras.callbacks import Callback
from generate import sample, generate

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

        diversity = 1.0
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

class Write_Text(Callback):
    def __init__(self, idx_to_char, char_to_idx, text, seq_len, vocab,
            filepath, gen_text_len=300, temperature=.5):
        self.gen_text_len = gen_text_len
        self.idx_to_char = idx_to_char
        self.char_to_idx = char_to_idx
        self.text = text
        self.seq_len = seq_len
        self.vocab = vocab
        self.filepath = filepath
        self.temperature = temperature

    def on_epoch_end(self, epoch, logs={}):
        generated = generate(model=self.model, idx_to_char=self.idx_to_char,
            char_to_idx=self.char_to_idx, gen_text_len=self.gen_text_len,
            seq_len=self.seq_len, vocab=self.vocab, text=self.text,
            temperature=self.temperature)

        try:
            filename = self.filepath.format(epoch=epoch)
            with open(filename, 'w') as f:
                f.write(generated)
            print("Saved composition to", filename)
        except Exception as e:
            print("Could not save", filename)
            print(e)
