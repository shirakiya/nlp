import os
import argparse

from nlp.dataset import get_files_list, file_generator
from nlp.preprocessing.cleaning import clean_text, clean_wiki
from nlp.preprocessing.tokenizer import Tokenizer
from nlp.preprocessing.normalization import normalize
from nlp.preprocessing.stopword import get_basic_stopwords_ja, remove_stopwords


def main(path):
    tokenizer = Tokenizer()
    stopwords = get_basic_stopwords_ja()

    files = get_files_list(path)
    generator = file_generator(files)
    for text in generator:
        text = clean_text(text)
        text = clean_wiki(text)
        words = tokenizer.wakati_baseform(text)
        words = [normalize(w) for w in words]
        words = remove_stopwords(words, stopwords)
        text = ' '.join(words)

        if text == '\n' or text == '':
            continue
        print(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        raise FileNotFoundError('Not found file or directory.')

    main(args.input_path)
