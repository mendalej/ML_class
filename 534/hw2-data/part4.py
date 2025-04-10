import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sys
import time
import pandas as pd
from svector import svector
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

def read_from(textfile):
    data = pd.read_csv(textfile)
    for _, row in data.iterrows():
        label = 1 if row['target'] == '+' else -1
        words = row['sentence'].split()
        yield label, words

# for part 3

def word_count(trainfile):
    word_counts = defaultdict(int)
    for i, words in read_from(trainfile):
        for word in words:
            word_counts[word] += 1
    return word_counts

def word_freq_filter(word_counts, min_count=1):
    freq_words = {word for word, count in word_counts.items() if count >= min_count}
    return freq_words
    

def make_vector(words, freq_words):
    v = svector()
    for word in words:
        if word in freq_words:
            v[word] += 1
    return v

# preparation of the data
def data_prep(trainfile, freq_words):
    freq_words = list(freq_words)
    x, y = [], []
    for label, words in read_from(trainfile):
        sent = make_vector(words, freq_words)
        vclength = np.zeros(len(freq_words))
        for i, word in enumerate(freq_words):
                if word in sent:
                    vclength[i] = sent[word]
        x.append(vclength)
        y.append(label)
    return np.array(x), np.array(y)

if __name__ == "__main__":
    t = time.time()
    new_word_counts = word_freq_filter(word_count(sys.argv[1]))
    xtrain, ytrain = data_prep(sys.argv[1], new_word_counts)
    new_model = LogisticRegression(max_iter=1000)
    new_model.fit(xtrain, ytrain)
    xdev, ydev = data_prep(sys.argv[2], new_word_counts)
    dev_pred = new_model.predict(xdev)
    dev_error = 1 - accuracy_score(ydev, dev_pred)
    print(f"Best dev error: {dev_error * 100:.1f}%", time.time() - t)
    test_data = pd.read_csv("test.csv")
    xtest, _ = data_prep("test.csv", new_word_counts)
    test_predictions = new_model.predict(xtest)
    test_data['target'] = np.where(test_predictions > 0, '+', '-')
    test_data.to_csv('test.predicted.csv', index=False)
