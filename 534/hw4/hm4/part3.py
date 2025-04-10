#!/usr/bin/env python3

import gensim
from gensim.models import KeyedVectors
import sys
import time
import pandas as pd
import numpy as np
from collections import defaultdict
# from xgboost import XGBClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neural_network import MLPClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

wv = KeyedVectors.load('embs_train.kv')
train_data = pd.read_csv('train.csv')
dev_data = pd.read_csv('dev.csv')
test_data = pd.read_csv('test.csv')

def word_count(sentences):
    word_counts = defaultdict(int)
    for sent in sentences:
        for word in sent.split():
            word_counts[word] += 1
    return word_counts

def pruned_words(wv, word_counts, min_count=1):
    freq_words = {word for word, count in word_counts.items() if count >= min_count}
    pruned_wv = {word: wv[word] for word in freq_words if word in wv}
    return pruned_wv

def sentence_embedding(sentence, pruned_wv):
    embedded = [pruned_wv[word] for word in sentence.split() if word in pruned_wv]
    if embedded:
        return np.mean(embedded, axis=0)
    else:
        return np.zeros(wv.vector_size)
    
word_counts = word_count(train_data['sentence'])

pruned_wv = pruned_words(wv, word_counts, min_count=1)

data_embedding = np.array([sentence_embedding(sentence, pruned_wv) for sentence in train_data['sentence']])
dev_embedding = np.array([sentence_embedding(sentence, pruned_wv) for sentence in dev_data['sentence']])
test_embedding = np.array([sentence_embedding(sentence, pruned_wv) for sentence in test_data['sentence']])
# train_labels = np.where(train_data['target'] == '+', 1, 0)
# dev_labels = np.where(dev_data['target'] == '+', 1, 0)
train_labels = train_data['target'].map({'+': 1, '-': -1})
dev_labels = dev_data['target'].map({'+': 1, '-': -1})


def data_prep(data_embedding, labels, dev_embedding, dev_labels):
    t = time.time()
    model = SVC(kernel='rbf')
    # model = DecisionTreeClassifier()
    # model = RandomForestClassifier(n_estimators=100)
    # model = GradientBoostingClassifier(n_estimators=100)
    # model = AdaBoostClassifier(n_estimators=100)
    # model = GaussianNB()
    # model = MLPClassifier() # good results
    # model = LogisticRegression()
    # model = XGBClassifier() # will not work with .map() as it is right now 
    
    model.fit(data_embedding, labels)
    dev_preds = model.predict(dev_embedding)
    dev_error = 1 - accuracy_score(dev_labels, dev_preds)
    print(f"Dev Error Rate: {dev_error * 100:.2f},  time: {time.time() - t:.1f} secs")
    return model
    

def predict(test_embedding, model):
    return model.predict(test_embedding)

if __name__ == "__main__":
    model = data_prep(data_embedding, train_labels, dev_embedding, dev_labels)
    test_predicts = predict(test_embedding, model)
    test_data['target'] = test_predicts
    test_data['target'] = test_data['target'].map({1: '+', -1: '-'})
    test_data.to_csv('prediction3.csv', index=False)
