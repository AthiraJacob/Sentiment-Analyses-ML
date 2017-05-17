from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import re
import itertools
from collections import Counter
# from utils import *

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def preprocess_sentence(sentence, stops):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    filtered_words = [w for w in tokens]
    return " ".join(filtered_words)


def preprocess(x):
    # stops = set(stopwords.words('enguplish'))
    X = []
    y = []
    for instances in x:
        # sentence = preprocess_sentence(instances[0], stops).split()
        sentence = clean_str(instances[0])
        # sentences.append(sentence)
        # sentence = sentence.split()
        if instances[1]*1 == 1:
        	sentiment = [1,0]
        else:
        	sentiment = [0,1] 

        X.append(sentence)
        y.append(sentiment)
    y = np.array(y)
    return X,y


def load_data(fold, split):
    with open(fold + 'CR.' + split) as f:
        data = []
        for line in f.readlines():
            if split != 'test':
                text, sentiment = line.split('\t')
                data.append([text.strip(), sentiment[0] == '1'])
            else:
                text = line
                data.append([text.strip(), False])
        return data


def load_data_and_labels(folder):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    train_data = load_data(folder, 'train')
    val_data = load_data(folder, 'dev')
    x_train,y_train = preprocess(train_data)
    x_val,y_val = preprocess(val_data)
    
    # positive_examples = list(open(positive_data_file, "r").readlines())
    # positive_examples = [s.strip() for s in positive_examples]
    # negative_examples = list(open(negative_data_file, "r").readlines())
    # negative_examples = [s.strip() for s in negative_examples]
    # # Split by words
    # x_text = positive_examples + negative_examples
    # x_text = [clean_str(sent) for sent in x_text]
    # # Generate labels
    # positive_labels = [[0, 1] for _ in positive_examples]
    # negative_labels = [[1, 0] for _ in negative_examples]
    # y = np.concatenate([positive_labels, negative_labels], 0)
    return x_train,y_train, x_val, y_val


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
