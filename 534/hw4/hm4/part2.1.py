# part 2.1 

import gensim
from gensim.models import KeyedVectors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score

train_data = pd.read_csv('train.csv')
dev_data = pd.read_csv('dev.csv')

# part 2.1
wv = KeyedVectors.load('embs_train.kv')

new = (wv['the'] + wv['man'] + wv['bit'] + wv['the'] + wv['dog']) / 5
# print("New embedding: ", new, "dtype:", new.dtype)

def sentence_embedding(sentence, wv):
    embedded = [wv[word] for word in sentence.split() if word in wv]
    if embedded:
        return np.mean(embedded, axis=0)
    else:
        return np.zeros(wv.vector_size)
    

data_embedding = np.array([sentence_embedding(sentence, wv) for sentence in train_data['sentence']])
dev_embedding = np.array([sentence_embedding(sentence, wv) for sentence in dev_data['sentence']])
labels = train_data['target']
dev_labels = dev_data['target']

def closest_sentence(emb_sent, data_emb):
    sim = cosine_similarity([emb_sent], data_emb)
    closest = np.argmax(sim)
    return closest

first_sentence = data_embedding[0]
# print(f"First sentence: {train_data['sentence'].iloc[0]} (Label: {train_data['target'].iloc[0]})")
idx_first = closest_sentence(first_sentence, data_embedding[1:]) + 1
first_closest = train_data['sentence'].iloc[idx_first]
# print(f"Closest sentence to first sentence: {train_data['sentence'].iloc[idx_first]} (Label: {train_data['target'].iloc[idx_first]})")

second_sentence = data_embedding[1]
# print(f"Second sentence: {train_data['sentence'].iloc[1]} (Label: {train_data['target'].iloc[1]})")
idx_second = closest_sentence(second_sentence, data_embedding[2:]) + 2
second_closest = train_data['sentence'].iloc[idx_second]
# print(f"Closest sentence to second sentence: {train_data['sentence'].iloc[idx_second]} (Label: {train_data['target'].iloc[idx_second]})")

# 2.3
train_errs = []
dev_errs = []
best_dev_err = float('inf')
best_k = None
for k in range(1, 100, 2):
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(data_embedding, labels)
    # predict on training set
    train_preds = knn.predict(data_embedding)
    train_err = 1 - accuracy_score(labels, train_preds)
    train_errs.append(train_err)
    # predict on dev set
    dev_preds = knn.predict(dev_embedding)
    dev_err = 1 - accuracy_score(dev_labels, dev_preds)
    dev_errs.append(dev_err)

    if dev_err < best_dev_err:
        best_dev_err = dev_err
        best_k = k
    
    print(f"k={k} train_err={train_err*100:.1f} dev_err={dev_err:.4f}")
print(f"Best dev error rate: {best_dev_err*100:.1f}% achieved with k={best_k}")

knn_best = KNeighborsClassifier(n_neighbors=best_k, metric='cosine')
knn_best.fit(data_embedding, labels)
dev_preds_best = knn_best.predict(dev_embedding)

prediction = pd.read_csv('test.csv')
prediction_embedding = np.array([sentence_embedding(sentence, wv) for sentence in prediction['sentence']])
n_neighbors = 53
prediction_preds = knn_best.predict(prediction_embedding)
prediction['target'] = prediction_preds
prediction.to_csv('prediction2.1.csv', index=False)







