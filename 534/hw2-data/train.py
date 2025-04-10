#!/usr/bin/env python3

# from __future__ import division # no need for python3, but just in case used w/ python2
# certain things have been commented for ease of reading the output

# this includes some of the code from part 2 and the code for part 3

import sys
import time
import pandas as pd
from svector import svector
from collections import defaultdict

def read_from(textfile):
    data = pd.read_csv(textfile)
    for i in range(len(data)):
        id, words, label = data.iloc[i]
        yield (1 if label=="+" else -1, words.split())

# for part 3

def word_count(trainfile):
    word_counts = defaultdict(int)
    for i, words in read_from(trainfile):
        for word in words:
            word_counts[word] += 1
    return word_counts

def word_freq_filter(word_counts, min_count=2):
    freq_words = {word for word, count in word_counts.items() if count >= min_count}
    return freq_words
    

def make_vector(words, freq_words):
    v = svector()
    for word in words:
        if word in freq_words:
            v[word] += 1
    v['bias'] = 1
    return v

def test(devfile, model, freq_words):
    tot, err = 0, 0
    for i, (label, words) in enumerate(read_from(devfile), 1):
        if label * (model.dot(make_vector(words, freq_words))) <= 0:
            err += 1
    return err / i  # Return error rate
    
            
def train(trainfile, devfile, epochs=10):
    t = time.time()
    best_err = 1.
    model = svector()
    # for part 3
    word_counts = word_count(trainfile)
    freq_words= word_freq_filter(word_counts)
    # for part 2 uses psuedocode from chapter 4 algorithm 7
    bias = 0
    avg_weight = svector()
    avg_bias = 0
    c = 1
    best_err = float('inf')
    for it in range(1, epochs+1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1): # label is +1 or -1
            sent = make_vector(words, freq_words)
            if label * (model.dot(sent)) <= 0:
                updates += 1
                model += label * sent
                # for part 2
                bias += label
                avg_weight += c * label * sent
                avg_bias += c * label
            c += 1
        avg_model = model - (1/c) * avg_weight
        avg_bias = bias - (1/c) * avg_bias
        dev_err = test(devfile, avg_model, freq_words)
        best_err = min(best_err, dev_err)
        print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    print("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(avg_model), time.time() - t))

    # for predictions that were made 
    def predictions(model, sentence):
        sent = make_vector(sentence.split(), freq_words)
        prediction_made = model.dot(sent)
        return '+' if prediction_made > 0 else '-'
    test_predict = pd.read_csv("test.csv")
    for index, row in test_predict.iterrows():
        if row['target'] == '?':
            prediction_made = predictions(avg_model, row['sentence'])
            test_predict.at[index, 'target'] = prediction_made
    test_predict.to_csv('test.predicted.csv', index=False)

if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2], 10)