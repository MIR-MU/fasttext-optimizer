import os
from io import open
import torch
import re

from wmt13 import read_words


# splitter = re.compile("[ \n\r\t\v\f\0]+")


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    train = None
    valid = None
    test = None

    def __init__(self, language, dictionary: Dictionary = None, create_datasets=True):
        self.language = language
        if dictionary is not None:
            self.dictionary = dictionary
        else:
            self.dictionary = Dictionary()
            if create_datasets:
                self.tokenize_all(create_dict=True)

    def tokenize_all(self, create_dict: bool):
        self.train = self.tokenize('train', create_dict)
        self.valid = self.tokenize('validation', create_dict)
        self.test = self.tokenize('test', create_dict)

    def tokenize(self, subset, create_dict: bool):
        """Tokenizes a text file."""

        language = self.language
        def words(): return read_words(language=language, subset=subset)

        # Add words to the dictionary
        if create_dict:
            for word in words():
                self.dictionary.add_word(word)

        # Tokenize file content
        ids = []
        for word in words():
            if word in self.dictionary.word2idx:
                ids.append(self.dictionary.word2idx[word])
        ids = torch.tensor(ids).type(torch.int64)

        return ids
