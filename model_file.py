import os
import time
import numpy as np

class ModelFile():
    def __init__(self, model_directory='', model='', weights='', history=''):
        self.model = model
        self.weights = weights
        self.history = history

        if(model_directory):
            self.model_directory = model_directory
            files = os.listdir(model_directory)
            if('model_history.npy' in files):
                self.history = os.path.join(model_directory, 'model_history.npy')
            if('model.h5' in files):
                self.model = os.path.join(model_directory, 'model.h5')
            weights_files = [f for f in files if 'weights-improvement' in f]
            if(weights_files):
                weights_files_split = [w.split('-') for w in weights_files ]
                epochs = [w[2] for w in weights_files_split]
                idx_max = np.argmax(epochs)
                self.weights = os.path.join(model_directory, \
                    weights_files[idx_max])
        else:
            timestr = time.strftime("%m%d-%H%M")
            model_name = "CharRNN_" + timestr
            self.model_directory = os.path.join('models', model_name)

    def check_exists(self):
        '''
        Returns true if model_directory, model, weights, and history all return
        true from os.path.exists()
        '''
        md = os.path.exists(self.model_directory)
        m = os.path.exists(self.model)
        w = os.path.exists(self.weights)
        h = os.path.exists(self.history)

        return md and m and w and h
