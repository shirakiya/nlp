import argparse
import os
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from nlp.preprocessing.cleaning import clean_text
from nlp.preprocessing.tokenizer import Tokenizer
from nlp.preprocessing.normalization import normalize
from nlp.preprocessing.stopword import get_basic_stopwords_ja, remove_stopwords
from nlp.word_vectors.bag_of_words import BagOfWords


def get_resource_path(path):
    dict_path = path + '.dict'
    corpus_path = path + '.mm'
    label_path = path + '_label.pkl'
    clf_path = path + '_clf.pkl'
    return dict_path, corpus_path, label_path, clf_path


def preprocessing(text, tokenizer, stopwords):
    text = clean_text(text)
    words = tokenizer.filter_by_pos(text)
    words = [normalize(w) for w in words]
    words = remove_stopwords(words, stopwords)
    return words


def load_livedoor_dataset(base_path):
    dict_path, corpus_path, label_path, _ = get_resource_path(base_path)

    bag_of_words = BagOfWords()
    bag_of_words.load_dictionary(dict_path)
    corpus = bag_of_words.load_corpus(corpus_path)
    vectors = bag_of_words.corpus_to_dense(corpus)

    with open(label_path, 'rb') as f:
        labels = np.array(pickle.load(f))

    print('complete loading dataset.')
    return vectors, labels


def top_k(result_list, k=5):
    x = np.array(result_list)
    argsort_index_r = np.argsort(x)[::-1]
    sort_value_r = np.sort(x)[::-1]

    index = []
    value = []
    for i in range(k):
        if i == k or i > len(sort_value_r):
            break
        index.append(argsort_index_r[i])
        value.append(sort_value_r[i])
    return index, value


def predict(base_path, text):
    dict_path, _, _, clf_path = get_resource_path(base_path)

    tokenizer = Tokenizer()
    stopwords = get_basic_stopwords_ja()
    bag_of_words = BagOfWords()
    bag_of_words.load_dictionary(dict_path)
    clf = joblib.load(clf_path)

    words = preprocessing(text, tokenizer, stopwords)
    feature = bag_of_words.to_dense(bag_of_words.doc2bow(words))
    result_p = clf.predict([feature])
    result_pp = clf.predict_proba([feature])

    print('predict:', result_p)
    print('predict_proba:', result_pp)
    print('top 5:', top_k(result_pp[0], k=5))


def train(base_path):
    print('collecting dataset.')
    x, y = load_livedoor_dataset(base_path)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

    print('train ...')
    clf = RandomForestClassifier(n_estimators=20, max_features='auto')
    clf.fit(x_train, y_train)

    score = clf.score(x_test, y_test)
    print('Score: ', score)

    clf_path = get_resource_path(base_path)[3]
    joblib.dump(clf, clf_path)
    print('model saved.')


def make_livedoor_dataset(source, exclude_files, base_path):
    dict_path, corpus_path, label_path = get_resource_path(base_path)
    tokenizer = Tokenizer()
    stopwords = get_basic_stopwords_ja()
    bag_of_words = BagOfWords()

    sentences = []
    label2id = {}
    labels = []
    dirs = [e for e in os.listdir(source) if os.path.isdir(os.path.join(source, e))]

    for root, _, files in os.walk(source):
        parent_dir = os.path.basename(root)
        if parent_dir in dirs and parent_dir not in label2id:
            label2id[parent_dir] = len(label2id)

        for file in files:
            if file in exclude_files:
                continue
            file_words = []
            for index, text in enumerate(open(os.path.join(root, file), 'r')):
                if index <= 1:
                    continue
                words = preprocessing(text, tokenizer, stopwords)
                file_words += words
            sentences.append(file_words)
            labels.append(label2id[parent_dir])

    bag_of_words.make_dictionary(sentences, dict_path)

    corpus = [bag_of_words.doc2bow(s) for s in sentences]
    bag_of_words.serialize_corpus(corpus, corpus_path)

    with open(label_path, 'wb') as f:
        pickle.dump(labels, f)

    print('Saved!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('base_path', type=str)
    parser.add_argument('-e', '--exclude-files-path', type=str, default='')
    args = parser.parse_args()

    # input_path = '/Users/shirakiya/datasets/nlp/livedoor-news-data/origin'
    # exclude_files_path = '/Users/shirakiya/datasets/nlp/livedoor-news-data/exclude_files.txt'
    # exclude_files = [f.strip() for f in open(exclude_files_path, 'r')]
    # base_path = 'data/livedoor'
    # text = 'iPadはどこに行けば変えますか？そしてそのiPadはどう使えばよいですか？'

    if not os.path.exists(args.input_path):
        raise FileNotFoundError('Not found file or directory.')

    exclude_files = []
    if args.exclude_files_path and os.path.isfile(args.exclude_files_path):
        exclude_files = [f.strip() for f in open(args.exclude_files_path, 'r')]

    make_livedoor_dataset(args.input_path, exclude_files, args.base_path)
    # train(base_path)
    # predict(base_path, text)
