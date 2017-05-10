import os
import urllib.request


def maybe_download(path):
    if os.path.exists(path):
        return
    url = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
    res = urllib.request.urlopen(url)
    body = res.read().decode('utf-8').strip()
    with open(path, 'w') as f:
        f.write(body)


def get_basic_stopwords_ja(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'stopwords.txt')
    maybe_download(path)

    with open(path, 'r') as f:
        stopwords = f.readlines()
    return [sw.strip() for sw in stopwords]


def remove_stopwords(words, stopwords=None):
    if stopwords is None:
        stopwords = get_basic_stopwords_ja()
    return [w for w in words if w not in stopwords]
