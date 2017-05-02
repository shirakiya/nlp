import argparse
from pprint import pprint

from nlp.word_vectors.word2vec import Word2Vec
from nlp.preprocessing.tokenizer import Tokenizer


def train(args):
    with open(args.input_path, 'r') as f:
        sentences = f.readlines()

    w = Word2Vec()
    w.train(sentences, args.model_path)


def get_vector(args):
    w = Word2Vec()
    w.load(args.model_path)

    t = Tokenizer()
    result = {}
    for word in t.wakati_baseform(args.text):
        try:
            result[word] = w.get_word_vector(word)
        except KeyError:
            result[word] = None

    pprint(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers()

    train_parser = sub_parser.add_parser('train')
    train_parser.add_argument('input_path', type=str)
    train_parser.add_argument('model_path', type=str)
    train_parser.set_defaults(func=train)

    vector_parser = sub_parser.add_parser('vector')
    vector_parser.add_argument('model_path', type=str)
    vector_parser.add_argument('text', type=str)
    vector_parser.set_defaults(func=get_vector)

    args = parser.parse_args()
    args.func(args)
