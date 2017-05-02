from gensim import corpora, matutils


class BagOfWords(object):

    def __init__(self):
        self._dictionary = None

    def make_dictionary(self, sentences, save_path):
        self._dictionary = corpora.Dictionary(sentences)
        self._dictionary.save(save_path)
        return self._dictionary

    def load_dictionary(self, load_path):
        self._dictionary = corpora.Dictionary.load(load_path)
        return self._dictionary

    def serialize_corpus(self, sentences, save_path):
        return corpora.MmCorpus.serialize(save_path, sentences)

    def load_corpus(self, load_path):
        return corpora.MmCorpus(load_path)

    def doc2bow(self, words):
        if self._dictionary is None:
            raise Exception('doc2bow() is available after make or load dictionary.')
        return self._dictionary.doc2bow(words)

    def to_dense(self, bow):
        if self._dictionary is None:
            raise Exception('to_dence() is available after make or load dictionary.')
        return list(matutils.corpus2dense([bow], num_terms=len(self._dictionary)).T[0])

    def corpus_to_dense(self, corpus):
        return [self.to_dense(bow) for bow in corpus]
