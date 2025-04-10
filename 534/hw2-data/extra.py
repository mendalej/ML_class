# this is an extra file that does the same thing as part4.py but has a different implementation
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
import time

def data_prep(textfile):
    # t = time.time()
    data = pd.read_csv(textfile)
    text_label = data['target'].apply(lambda x: 1 if x == '+' else -1)
    sent= data['sentence']
    # print(time.time() - t)
    return sent, text_label

def error(y, y_pred):
    return 1 - accuracy_score(y, y_pred)

if __name__ == "__main__":
    xtrain, ytrain = data_prep("train.csv")
    vec = TfidfVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))
    xtrain_tvec = vec.fit_transform(xtrain)
    param_grid = {
    'C': [0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear'], 
    'class_weight': ['balanced']
    }
    # param_grid = {'C': [0.1, 1, 10, 100]}
    model = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
    model.fit(xtrain_tvec, ytrain)
    x_dev, y_dev = data_prep("dev.csv")
    x_dev_tvec = vec.transform(x_dev)
    d_pred = model.predict(x_dev_tvec)
    dev_error = 1 - accuracy_score(y_dev, d_pred)
    print(f"Dev error: {dev_error * 100:.1f}")
    test_data = pd.read_csv("test.csv")
    x_test, _ = data_prep("test.csv")
    x_test_tvec = vec.transform(x_test)
    test_pred = model.predict(x_test_tvec)
    test_data['target'] = np.where(test_pred == 1, '+', '-')
    test_data.to_csv('test.predicted.csv', index=False)