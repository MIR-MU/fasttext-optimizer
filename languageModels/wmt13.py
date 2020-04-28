from itertools import chain
from multiprocessing import Pool
from pathlib import Path
import re

from gensim.utils import tokenize as _gensim_tokenize
from tqdm import tqdm as _tqdm_tqdm


FILENAMES = {
    'ar': {
        'validation': ('../data/wmt13/training/news-commentary-v8.ar', 159873),
    },
    'cs': {
        'train': ('../data/wmt13/training-monolingual/news.2011.cs.shuffled', 8746448),
        'validation': ('../data/wmt13/training/news-commentary-v8.cs', 162309),
        'test': ('../data/wmt13/training-monolingual/news.2012.cs.shuffled', 7538499),
    },
    'de': {
        'train': ('../data/wmt13/training-monolingual/news.2011.de.shuffled', 16037788),
        'validation': ('../data/wmt13/training/news-commentary-v8.de', 204276),
        'test': ('../data/wmt13/training-monolingual/news.2012.de.shuffled', 20673844),
    },
    'en': {
        'train': ('../data/wmt13/training-monolingual/news.2011.en.shuffled', 15437674),
        'validation': ('../data/wmt13/training/news-commentary-v8.en', 247966),
        'test': ('../data/wmt13/training-monolingual/news.2012.en.shuffled', 14869673),
    },
    'es': {
        'train': ('../data/wmt13/training-monolingual/news.2011.es.shuffled', 5691123),
        'validation': ('../data/wmt13/training/news-commentary-v8.es', 206534),
        'test': ('../data/wmt13/training-monolingual/news.2012.es.shuffled', 4189396),
    },
    'fr': {
        'train': ('../data/wmt13/training-monolingual/news.2011.fr.shuffled', 6030152),
        'validation': ('../data/wmt13/training/news-commentary-v8.fr', 193714),
        'test': ('../data/wmt13/training-monolingual/news.2012.fr.shuffled', 4114360),
    },
    'ru': {
        'train': ('../data/wmt13/training-monolingual/news.2011.ru.shuffled', 9945918),
        'validation': ('../data/wmt13/training/news-commentary-v8.ru', 183083),
        'test': ('../data/wmt13/training-monolingual/news.2012.ru.shuffled', 9789861),
    },
}


def _simple_preprocess(text):
    tokens = [
        token
        for token
        in _gensim_tokenize(text, lower=True, deacc=False, errors='ignore')
        if not token.startswith('_')
    ]
    return tokens


def _tqdm(*args, **kwargs):
    return _tqdm_tqdm(*args, **{**kwargs, **{'miniters': 10000}})


def read_words(subset='validation', language='cs', percentage=0.1):
    filename, num_lines = FILENAMES[language][subset]
    filename = Path(filename)
    num_lines = int(num_lines)
    effective_num_lines = int(percentage * num_lines)
    with filename.open('rt') as f:
        with Pool(None) as pool:
            lines = _tqdm(f, desc='Reading {}'.format(filename.name), total=effective_num_lines)
            lines = (line for line, _ in zip(lines, range(effective_num_lines)))
            for line in pool.imap(_simple_preprocess, lines):
                for word in line:
                    yield word
