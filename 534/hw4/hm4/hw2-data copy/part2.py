#!/usr/bin/env python3
# this is the code for part 2 of the assignment 
# I have used the psuedocode from chapter 4 algorithm 7
# train.py contains the code for part 3 and some of the code for part 2 part 1
# I decided to remove the code for part 2 questions 3 and 4 just for visiability (it was annoying me in train)

from __future__ import division # no need for python3, but just in case used w/ python2

import sys
import time
import pandas as pd
from svector import svector

def read_from(textfile):
    data = pd.read_csv(textfile)
    for i in range(len(data)):
        id, words, label = data.iloc[i]
        yield (1 if label=="+" else -1, words.split())

def make_vector(words):
    v = svector()
    for word in words:
        v[word] += 1
    v['<bias>'] = 1
    return v
            
def train(trainfile, devfile, epochs=10):
    t = time.time()
    best_err = float('inf')
    model = svector()
    bias = 0
    avg_weight = svector()
    best_model = svector()
    avg_bias = 0
    c = 1
    for it in range(1, epochs+1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1): # label is +1 or -1
            sent = make_vector(words)
            if label * (model.dot(sent)) <= 0:
                updates += 1
                model += label * sent
                bias += label
                avg_weight += c * label * sent
                avg_bias += c * label
                updates += 1
            c += 1   
        avg_model = model - (1/c) * avg_weight
        avg_bias = bias - (1/c) * avg_bias
        dev_err = test(devfile, avg_model)
        if devfile:
            if dev_err < best_err:
                best_err = dev_err
                best_model = avg_model
        print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    print("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(avg_model), time.time() - t))
    return best_model
def test(devfile, model):
    tot, err = 0, 0
    for i, (label, words) in enumerate(read_from(devfile), 1):
        if label * (model.dot(make_vector(words))) <= 0:
            err += 1
    return err / i  # Return error rate

def top_features(model, n=20):
    # for part 2 q 3
    top_20 = sorted(model.items(), key=lambda x: x[1], reverse=True)
    top_positive = top_20[:n]
    top_negative = top_20[-n:]
    print("Top 20 Positive Features:")
    for i, weight in top_positive:
        print(f"{i}: {weight:.4f}")
    print("Top 20 Negative Features:")
    for i, weight in top_negative:
        print(f"{i}: {weight:.4f}")
    return top_positive, top_negative

def missclassified(model, devfile):
    # for part 2 q 4
    pos_false = []
    neg_false = []
    for label, words in read_from(devfile): # note 1...|D|
        # for part 2 q 4
        vector = make_vector(words)
        confidence = (model.dot(vector))
        prediction = '+' if confidence > 0 else '-'
        if label * model.dot(vector) <= 0:
            if label == - 1 and prediction == '+':
                pos_false.append((confidence, words))
            elif label == 1 and prediction == '-':
                neg_false.append((confidence, words))
    # for part 2 q 4
    pos_false = sorted(pos_false, key=lambda x: abs(x[0]), reverse=True)[:5]
    neg_false = sorted(neg_false, key=lambda x: abs(x[0]), reverse=True)[:5]
    print("Top 5 Negative examples shown as Positive:")
    for confidence, words in pos_false:
        print(f"Score: {confidence:.4f}, Words: {' '.join(words)}")
    print("Top 5 Positive examples shown as negative:")
    for confidence, words in neg_false:
        print(f"Score: {confidence:.4f}, Words: {' '.join(words)}")
    return pos_false, neg_false

def predictions(model, sentence):
    sent = make_vector(sentence.split())
    prediction_made = model.dot(sent)
    return '+' if prediction_made > 0 else '-'

if __name__ == "__main__":
    new_model =  train(sys.argv[1], sys.argv[2], 10)
    top_positive, top_negative = top_features(new_model, n=20)
    pos_false, neg_false = missclassified(new_model, 'dev.csv')
    test_predict = pd.read_csv("test.csv")
    for index, row in test_predict.iterrows():
        if row['target'] == '?':
            prediction_made = predictions(new_model, row['sentence'])
            test_predict.at[index, 'target'] = prediction_made
    test_predict.to_csv('test.predicted.csv', index=False)


