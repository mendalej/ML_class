#!/usr/bin/env python3

import gensim
from gensim.models import KeyedVectors
import sys
import time
import pandas as pd
import numpy as np
from collections import defaultdict
from svector import svector

wv = KeyedVectors.load('embs_train.kv')
train_data = pd.read_csv('train.csv')
dev_data = pd.read_csv('dev.csv')
test_data = pd.read_csv('test.csv')

# word count and word filtering and pruning

def word_count(sentences):
    word_counts = defaultdict(int)
    for sent in sentences:
        for word in sent.split():
            word_counts[word] += 1
    return word_counts

def word_freq_filter(sentences, word_counts, min_count=2):
    freq_words = {word for word, count in word_counts.items() if count >= min_count}
    filtered_sentences = [' '.join([word for word in sent.split() if word in freq_words]) for sent in sentences]
    return filtered_sentences

def pruned_words(wv, word_counts, min_count=2):
    freq_words = {word for word, count in word_counts.items() if count >= min_count}
    pruned_wv = {word: wv[word] for word in freq_words if word in wv}
    return pruned_wv

# sentence embedding for word2vec and the embedding and labels that go with it
def sentence_embedding(sentence, pruned_wv):
    embedded = [pruned_wv[word] for word in sentence.split() if word in pruned_wv]
    if embedded:
        return np.mean(embedded, axis=0)
    else:
        return np.zeros(wv.vector_size)
    
word_counts = word_count(train_data['sentence'])
train_data['sentence'] = word_freq_filter(train_data['sentence'], word_counts)
dev_data['sentence'] = word_freq_filter(dev_data['sentence'], word_counts)
test_data['sentence'] = word_freq_filter(test_data['sentence'], word_counts)

pruned_wv = pruned_words(wv, word_counts, min_count=1)

data_embedding = np.array([sentence_embedding(sentence, pruned_wv) for sentence in train_data['sentence']])
dev_embedding = np.array([sentence_embedding(sentence, pruned_wv) for sentence in dev_data['sentence']])
test_embedding = np.array([sentence_embedding(sentence, pruned_wv) for sentence in test_data['sentence']])
labels = train_data['target'].map({'+': 1, '-': -1})
dev_labels = dev_data['target'].map({'+': 1, '-': -1})

# one hot encoding
def one_hot_encoding(sentence, vocab):
    v = np.zeros(len(vocab))
    for word in sentence.split():
        if word in vocab:
            v[vocab[word]] += 1
    return  v

# one hot labels
vocab = {word: idx for idx, word in enumerate(word_counts.keys())}
train_one_hot = np.array([one_hot_encoding(sentence, vocab) for sentence in train_data['sentence']])
dev_one_hot = np.array([one_hot_encoding(sentence, vocab) for sentence in dev_data['sentence']])

train_one_hot_labels = train_data['target'].map({'+': 1, '-': -1}).to_numpy()
dev_one_hot_labels = dev_data['target'].map({'+': 1, '-': -1}).to_numpy()


# training and testing 
def train(data_embedding, labels, dev_embedding, dev_labels, epochs=10):
    t = time.time()
    best_err = 1.0
    model = np.zeros(data_embedding.shape[1])
    bias = 0
    avg_weight = np.zeros(data_embedding.shape[1])
    best_model = np.zeros(data_embedding.shape[1])
    avg_bias = 0
    c = 1
    for it in range(1, epochs+1):
        updates = 0
        for i, (label, features) in enumerate(zip(labels, data_embedding), 1):
            predict = np.dot(model, features) + bias
            if label * predict <= 0:
                updates += 1
                model += label * features
                bias += label
                avg_weight += c * label * features
                avg_bias += c * label
            c += 1
        avg_model = model - (avg_weight / c)
        avg_bias = bias - (avg_bias / c)
        dev_err = test(dev_embedding, dev_labels, avg_model)
        if dev_err < best_err:
            best_err = dev_err
            best_model = avg_model
    #     print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    # print("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(avg_model), time.time() - t))
    return best_model

# comparison
def compare(dev_data, dev_embedding, dev_one_hot, dev_labels, word2vec_model, one_hot_model):
    word2vec_preds = np.sign(np.dot(dev_embedding, word2vec_model))
    one_hot_preds = np.sign(np.dot(dev_one_hot, one_hot_model))

    for i in range(len(dev_data)):
        sentence = dev_data.iloc[i]['sentence']
        true_label = dev_labels[i]
        wv_pred = word2vec_preds[i]
        oh_pred = one_hot_preds[i]

        # Check for positive and negative cases
        if true_label == 1 and wv_pred == 1 and oh_pred == -1:
            print(f"Example {i}: Correct with Word2Vec, Incorrect with One-Hot")
            print(f"Sentence: {sentence}")
            print(f"True Label: {true_label}, Word2Vec: {wv_pred}, One-Hot: {oh_pred}\n")
        elif true_label == -1 and wv_pred == -1 and oh_pred == 1:
            print(f"Example {i}: Correct with Word2Vec, Incorrect with One-Hot")
            print(f"Sentence: {sentence}")
            print(f"True Label: {true_label}, Word2Vec: {wv_pred}, One-Hot: {oh_pred}\n")
               

def test(dev_embedding, dev_labels, model):
    tot, err = 0, 0
    for i, (label, features) in enumerate(zip(dev_labels, dev_embedding), 1):
        predict = np.dot(model, features)
        err += (label * predict) <= 0
    return err / i  # Return error rate

def predict(test_embedding, model):
    predicts = np.sign(np.dot(test_embedding, model))
    return predicts

if __name__ == "__main__":
    train_labels = train_data['target'].map({'+': 1, '-': -1})
    dev_labels = dev_data['target'].map({'+': 1, '-': -1})
    model = train(data_embedding, train_labels, dev_embedding, dev_labels)
    word2vec_feats = train(data_embedding, labels, dev_embedding, dev_labels, epochs=10)
    one_hot_feats = train(train_one_hot, train_one_hot_labels, dev_one_hot, dev_one_hot_labels, epochs=10)
    compare(dev_data, dev_embedding, dev_one_hot, dev_labels, word2vec_feats, one_hot_feats)

    test_predicts = predict(test_embedding, model)
    test_data['target'] = test_predicts
    test_data['target'] = test_data['target'].map({1: '+', -1: '-'})
    test_data.to_csv('prediction2.2q5.csv', index=False)
