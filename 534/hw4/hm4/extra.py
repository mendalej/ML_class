import sys
import time
import pandas as pd
from svector import svector
import gensim
from gensim.models import KeyedVectors


wv = KeyedVectors.load('embs_train.kv')
train_data = pd.read_csv('train.csv')
dev_data = pd.read_csv('dev.csv')
test_data = pd.read_csv('test.csv')

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


def predictions(model, sentence):
    sent = make_vector(sentence.split())
    prediction_made = model.dot(sent)
    return '+' if prediction_made > 0 else '-'

if __name__ == "__main__":
    new_model =  train(sys.argv[1], sys.argv[2], 10)
    test_predict = pd.read_csv("test.csv")
    for index, row in test_predict.iterrows():
        if row['target'] == '?':
            prediction_made = predictions(new_model, row['sentence'])
            test_predict.at[index, 'target'] = prediction_made
    test_predict.to_csv('test.predicted.csv', index=False)


