# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import numpy as np
from math import sqrt
import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict



    
def scaleX(X):
    """
    unit length scaling of each example
    """
    Xscaled=np.copy(X);
    for i in range(Xscaled.shape[0]):
        Xscaled[i] = Xscaled[i]/sqrt(np.sum(Xscaled[i]**2))
    return Xscaled
    
def toXy(data, key_to_ix):
    X = np.vstack([textToFeature(txt, key_to_ix) for txt, _ in data])
    y = np.array([int(sentiment) * 2 - 1 for _, sentiment in data])
    return X, y

def textToFeature(text, key_to_ix):
    words = text.split()
    feat = np.zeros((len(key_to_ix)))
    akey = next (iter (key_to_ix.keys()))
    if isinstance(akey, tuple): # bigram
        for i in range(len(words) - 1):
            if (words[i], words[i + 1]) in key_to_ix:
                feat[key_to_ix[(words[i], words[i + 1])]] = 1
                #feat[key_to_ix[(words[i], words[i + 1])]] += 1
        
    else:
        for word in words:
            if word in key_to_ix:
                feat[key_to_ix[word]] = 1
                #feat[key_to_ix[word]] += 1 # if we want actual counts, not just present/absent
    
    return feat
    
def getVocab(data, use_bigram,mincount):
    """
    construct vocabulary from data
    only include terms that appear at least mincount times
    """
    word_count = {}
    if use_bigram:
        for text, _ in data:
            words = text.split()
            for i in range(len(words) - 1):
                word_count[(words[i], words[i + 1])] = word_count.get((words[i], words[i + 1]), 0) + 1
    else:
        for text, _ in data:
            for word in text.split():
                word_count[word] = word_count.get(word, 0) + 1
                
    for word in list(word_count):
        if word_count[word] < mincount:
            del word_count[word]
    print("Feature size: ", len(word_count))
    # print(word_count.keys())
    keys = word_count.keys()
    key_to_ix = {key:ix for ix, key in enumerate(keys)}
    
    return keys, key_to_ix

class MeanEmbeddingVectorizer():
    def __init__(self, word2vec, size):
        self.word2vec = word2vec

        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = size

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])



class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec, size):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = size

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


def preprocess_sentence(sentence, stops):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    filtered_words = [w for w in tokens]
    return " ".join(filtered_words)


def preprocess(x):
    stops = set(stopwords.words('english'))
    X = []
    y = []
    for instances in x:
        sentence = preprocess_sentence(instances[0], stops).split()
        sentiment = instances[1]*1
        X.append(sentence)
        y.append(sentiment)
    return X,y
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from math import sqrt
import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict


def load_data(fold, split):
    with open(fold + '/CR.' + split) as f:
        data = []
        for line in f.readlines():
            if split != 'test':
                text, sentiment = line.split('\t')
                data.append([text.strip(), sentiment[0] == '1'])
            else:
                text = line
                data.append([text.strip(), False])
        return data
    
def scaleX(X):
    """
    unit length scaling of each example
    """
    Xscaled=np.copy(X);
    for i in range(Xscaled.shape[0]):
        Xscaled[i] = Xscaled[i]/sqrt(np.sum(Xscaled[i]**2))
    return Xscaled
    
def toXy(data, key_to_ix):
    X = np.vstack([textToFeature(txt, key_to_ix) for txt, _ in data])
    y = np.array([int(sentiment) * 2 - 1 for _, sentiment in data])
    return X, y

def textToFeature(text, key_to_ix):
    words = text.split()
    feat = np.zeros((len(key_to_ix)))
    akey = next (iter (key_to_ix.keys()))
    if isinstance(akey, tuple): # bigram
        for i in range(len(words) - 1):
            if (words[i], words[i + 1]) in key_to_ix:
                feat[key_to_ix[(words[i], words[i + 1])]] = 1
                #feat[key_to_ix[(words[i], words[i + 1])]] += 1
        
    else:
        for word in words:
            if word in key_to_ix:
                feat[key_to_ix[word]] = 1
                #feat[key_to_ix[word]] += 1 # if we want actual counts, not just present/absent
    
    return feat
    
def getVocab(data, use_bigram,mincount):
    """
    construct vocabulary from data
    only include terms that appear at least mincount times
    """
    word_count = {}
    if use_bigram:
        for text, _ in data:
            words = text.split()
            for i in range(len(words) - 1):
                word_count[(words[i], words[i + 1])] = word_count.get((words[i], words[i + 1]), 0) + 1
    else:
        for text, _ in data:
            for word in text.split():
                word_count[word] = word_count.get(word, 0) + 1
                
    for word in list(word_count):
        if word_count[word] < mincount:
            del word_count[word]
    print("Feature size: ", len(word_count))
    # print(word_count.keys())
    keys = word_count.keys()
    key_to_ix = {key:ix for ix, key in enumerate(keys)}
    
    return keys, key_to_ix

class MeanEmbeddingVectorizer():
    def __init__(self, word2vec, size):
        self.word2vec = word2vec

        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = size

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])



class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec, size):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = size

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])



