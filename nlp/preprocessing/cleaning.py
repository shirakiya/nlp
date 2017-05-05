import re


def clean_text_en(text):
    '''
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    '''
    text = re.sub(r'[^A-Za-z0-9(),!?\'\`]', ' ', text)
    text = re.sub(r'\'s', ' \'s', text)
    text = re.sub(r'\'ve', ' \'ve', text)
    text = re.sub(r'n\'t', ' n\'t', text)
    text = re.sub(r'\'re', ' \'re', text)
    text = re.sub(r'\'d', ' \'d', text)
    text = re.sub(r'\'ll', ' \'ll', text)
    text = re.sub(r',', ' , ', text)
    text = re.sub(r'!', ' ! ', text)
    text = re.sub(r'\(', ' \( ', text)
    text = re.sub(r'\)', ' \) ', text)
    text = re.sub(r'\?', ' \? ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip().lower()


def clean_text_ja(text):
    text = text.lower()  # 全て小文字に変換
    text = re.sub(r'[【】]', ' ', text)       # 【】の除去
    text = re.sub(r'[（）()]', ' ', text)     # （）の除去
    text = re.sub(r'[［］\[\]]', ' ', text)   # ［］の除去
    text = re.sub(r'[@＠]\w+', '', text)  # メンションの除去
    text = re.sub(r'https?:\/\/.*?[\r\n ]', '', text)  # URLの除去
    text = re.sub(r'　', ' ', text)  # 全角空白の除去
    return text


def clean_wiki(text):
    return re.sub(r'</?doc.*\n?', '', text)
