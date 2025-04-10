import gensim
from gensim.models import KeyedVectors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from svector import svector
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer


train_data = pd.read_csv('train.csv')
dev_data = pd.read_csv('dev.csv')

# part 2.1
wv = KeyedVectors.load('embs_train.kv')

vectorizer = CountVectorizer()


train_vec = vectorizer.fit_transform(train_data['sentence']).toarray()
dev_vec = vectorizer.transform(dev_data['sentence']).toarray()


labels = train_data['target']
dev_labels = dev_data['target']

train_errs = []
dev_errs = []
best_dev_err = float('inf')
best_k = None
for k in range(1, 100, 2):
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(train_vec, labels)
    # predict on training set
    train_preds = knn.predict(train_vec)
    train_err = 1 - accuracy_score(labels, train_preds)
    train_errs.append(train_err)
    # predict on dev set
    dev_preds = knn.predict(dev_vec)
    dev_err = 1 - accuracy_score(dev_labels, dev_preds)
    dev_errs.append(dev_err)

    if dev_err < best_dev_err:
        best_dev_err = dev_err
        best_k = k
    
    print(f"k={k} train_err={train_err*100:.1f} dev_err={dev_err:.4f}")
print(f"Best dev error rate: {best_dev_err*100:.1f}% achieved with k={best_k}")

knn_best = KNeighborsClassifier(n_neighbors=best_k, metric='cosine')
knn_best.fit(train_vec, labels)
dev_preds_best = knn_best.predict(dev_vec)
prediction = pd.read_csv('test.csv')
prediction_vec = vectorizer.transform(prediction['sentence']).toarray()
n_neighbors = 11
prediction_preds = knn_best.predict(prediction_vec)
prediction['target'] = prediction_preds
prediction.to_csv('prediction2.1q4.csv', index=False)