import os
from gensim.models import word2vec


class Word2Vec(object):

    default_params = {
        'sg': 1,  # 0: CBOW, 1: skip-gram
        'alpha': 0.025,
        'min_alpha': 0.0001,
        'size': 300,
        'window': 5,
        'min_count': 5,
        'workers': os.cpu_count(),
        'iter': 5,
    }

    def __init__(self):
        self._model = None

    def train(self, sentences, save_path, **params):
        train_params = {}
        for k, v in self.default_params.items():
            train_params[k] = params.get(k, v)

        self._model = word2vec.Word2Vec(sentences, **train_params)
        self._model.save(save_path)

    def load(self, load_path):
        self._model = word2vec.Word2Vec.load(load_path)

    def get_word_vector(self, word):
        if self._model is None:
            raise Exception('get_word_vector() is available after make or load dictionary.')
        try:
            vector = self._model.wv[word]
        except KeyError:
            raise KeyError("Word doesn't exists.")
        return vector
