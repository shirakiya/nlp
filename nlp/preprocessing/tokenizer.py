import os
import shlex
import subprocess
from collections import namedtuple
import MeCab


class Tokenizer(object):

    neologd_filename = 'mecab-ipadic-neologd'

    def __init__(self, use_neologd=True):
        mecab_dicdir = self.__dicdir()
        if use_neologd and os.path.exists(os.path.join(mecab_dicdir, self.neologd_filename)):
            option = '-d {}/{}'.format(mecab_dicdir, self.neologd_filename)
        else:
            option = '-d {}/ipadic'.format(mecab_dicdir)

        self.mecab = MeCab.Tagger(option)

    def __dicdir(self):
        result = subprocess.run(shlex.split('mecab-config --dicdir'),
                                stdout=subprocess.PIPE,
                                universal_newlines=True)
        return result.stdout.strip()

    def wakati(self, text):
        return [token.surface for token in self.tokenize(text)]

    def wakati_baseform(self, text):
        return [token.base_form if token.base_form != '*' else token.surface
                for token in self.tokenize(text)]

    def tokenize(self, text):
        token = namedtuple('Token', [
            'surface',
            'pos',
            'pos_detail1',
            'pos_detail2',
            'pos_detail3',
            'infl_type',
            'infl_form',
            'base_form',
            'reading',
            'phonetic'
        ])
        chunks = self.mecab.parse(text).splitlines()[:-1]  # to remove 'EOS'

        for chunk in chunks:
            if chunk == '':
                continue
            surface, feature = chunk.split('\t')
            features = feature.split(',')
            if len(features) <= 7:  # 読みが無い
                features.append('')
            if len(features) <= 8:  # 発音が無い
                features.append('')
            yield token(surface, *features)
