import os
import argparse

from nlp.dataset import get_files_list, file_generator
from nlp.preprocessing.cleaning import clean_text, clean_wiki
from nlp.preprocessing.tokenizer import Tokenizer
from nlp.preprocessing.normalization import normalize
from nlp.preprocessing.stopword import get_basic_stopwords_ja, remove_stopwords


def main(input_path, output_path, processing_flags):
    tokenizer = Tokenizer()
    stopwords = get_basic_stopwords_ja()
    output_file = open(output_path, 'a')

    files = get_files_list(input_path)
    for text in file_generator(files):
        if processing_flags['clean']:
            text = clean_text(text)
        if processing_flags['wiki']:
            text = clean_wiki(text)

        words = tokenizer.wakati_baseform(text)

        if processing_flags['normalize']:
            words = [normalize(w) for w in words]
        if processing_flags['stopword']:
            words = remove_stopwords(words, stopwords)

        text = ' '.join(words)

        if text == '\n' or text == '':
            continue
        else:
            output_file.write(f'{text}\n')

    output_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('-c', '--clean', action='store_true')
    parser.add_argument('-w', '--wiki', action='store_true')
    parser.add_argument('-n', '--normalize', action='store_true')
    parser.add_argument('-s', '--stopword', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        raise FileNotFoundError('Not found file or directory.')

    # overwritten output contents
    if os.path.exists(args.output_path):
        os.remove(args.output_path)

    processing_flags = {
        'clean': args.clean,
        'wiki': args.wiki,
        'normalize': args.normalize,
        'stopword': args.stopword,
    }
    main(args.input_path, args.output_path, processing_flags)
