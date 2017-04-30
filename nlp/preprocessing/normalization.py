import re
import unicodedata


def normalize_unicode(text, form='NFKC'):
    return unicodedata.normalize(form, text)


def normalize_number(text):
    return re.sub(r'\d+', '0', text)


def normalize_lower(text):
    return text.lower()


def normalize(text):
    normalized_text = normalize_unicode(text)
    normalized_text = normalize_number(normalized_text)
    normalized_text = normalize_lower(normalized_text)
    return normalized_text
