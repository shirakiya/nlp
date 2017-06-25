import argparse
import os
import pickle
import numpy as np
import base_import  # noqa
from nlp.datasets.livedoor import Livedoor
from nlp.preprocessing.cleaning import clean_text_ja
from nlp.preprocessing.tokenizer import Tokenizer
from nlp.preprocessing.normalization import normalize
from nlp.preprocessing.stopword import get_basic_stopwords_ja, remove_stopwords
from nlp.word_vectors.bag_of_words import BagOfWords
from nlp.learning.random_forest import RandomForest


def get_resource_path(path):
    dict_path = path + '.dict'
    corpus_path = path + '.mm'
    label_path = path + '_label.pkl'
    return dict_path, corpus_path, label_path


def preprocessing(text, tokenizer, stopwords):
    text = clean_text_ja(text)
    words = tokenizer.filter_by_pos(text)
    words = [normalize(w) for w in words]
    words = remove_stopwords(words, stopwords)
    return words


def load_livedoor_dataset(base_path):
    dict_path, corpus_path, label_path = get_resource_path(base_path)

    bag_of_words = BagOfWords()
    bag_of_words.load_dictionary(dict_path)
    corpus = bag_of_words.load_corpus(corpus_path)
    vectors = bag_of_words.corpus_to_dense(corpus)

    with open(label_path, 'rb') as f:
        labels = np.array(pickle.load(f))

    print('complete loading dataset.')
    return vectors, labels


def make_livedoor_dataset(args):
    base_path = args.base_path
    source = args.source
    if not os.path.exists(source):
        raise FileNotFoundError('Not found file or directory.')

    dict_path, corpus_path, label_path = get_resource_path(base_path)
    tokenizer = Tokenizer()
    stopwords = get_basic_stopwords_ja()

    sentences = []

    livedoor = Livedoor(source)
    texts, labels = livedoor.get_data()

    for text in texts:
        sentence = []
        for l in text.split('\n'):
            sentence += preprocessing(l, tokenizer, stopwords)
        sentences.append(sentence)

    bag_of_words = BagOfWords()
    bag_of_words.make_dictionary(sentences, dict_path)

    corpus = [bag_of_words.doc2bow(s) for s in sentences]
    bag_of_words.serialize_corpus(corpus, corpus_path)

    with open(label_path, 'wb') as f:
        pickle.dump(labels, f)

    print('Saved!')


def train(args):
    base_path = args.base_path
    model_path = args.model_path
    with_grid_search = args.with_grid_search

    print('collecting dataset.')
    x, y = load_livedoor_dataset(base_path)

    trainer = RandomForest(model_path)
    if with_grid_search:
        trainer.train_with_gridsearch(x, y)
    else:
        trainer.train(x, y)

    print('finish.')


def predict(args):
    base_path = args.base_path
    model_path = args.model_path
    text = args.text

    dict_path = get_resource_path(base_path)[0]

    tokenizer = Tokenizer()
    stopwords = get_basic_stopwords_ja()
    bag_of_words = BagOfWords()
    bag_of_words.load_dictionary(dict_path)
    feature = bag_of_words.to_dense(
        bag_of_words.doc2bow(preprocessing(text, tokenizer, stopwords)))

    estimator = RandomForest(model_path)
    result_p = estimator.predict([feature])
    result_pp = estimator.predict_proba([feature])
    result_ppk = estimator.predict_proba_top_k([feature])

    print('predict:', result_p)
    print('predict_proba:', result_pp)
    print('top 5:', result_ppk)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers()

    dataset_parser = sub_parser.add_parser('dataset')
    dataset_parser.add_argument('base_path', type=str)
    dataset_parser.add_argument('source', type=str)
    dataset_parser.set_defaults(func=make_livedoor_dataset)

    train_parser = sub_parser.add_parser('train')
    train_parser.add_argument('base_path', type=str)
    train_parser.add_argument('model_path', type=str)
    train_parser.add_argument('-g', '--with-grid-search', action='store_true')
    train_parser.set_defaults(func=train)

    predict_parser = sub_parser.add_parser('predict')
    predict_parser.add_argument('base_path', type=str)
    predict_parser.add_argument('model_path', type=str)
    predict_parser.add_argument('text', type=str)
    predict_parser.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)
