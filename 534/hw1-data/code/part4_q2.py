import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score


# code copied from part 3
train_data = pd.read_csv('income.train.5k.csv')
dev_data = pd.read_csv('income.dev.csv')

# train_feats = train_data.drop(columns=['target'])
# dev_feats = dev_data.drop(columns=['target'])

target_train = train_data['target']
target_dev = dev_data['target']

num_processor = MinMaxScaler(feature_range=(0, 1))
cat_processor = OneHotEncoder(sparse_output=False, handle_unknown='ignore')


preprocessor = ColumnTransformer([
    ('num', num_processor, ['age', 'hours']),
    ('cat', cat_processor, ['sector', 'edu', 'marriage', 'occupation', 'race', 'sex', 'country'])
])

binary_train = preprocessor.fit_transform(train_data)
binary_dev = preprocessor.transform(dev_data) 

# knn implementation starts with both euclidean and manhattan_equations
# def euclidean_equation(a, b):
#     return np.sum((a - b) ** 2, axis=1)

def manhattan_equation(a, b):
    return np.sum(np.abs(a - b), axis=1)

# def euclidean_equation(a, b):
#     return np.linalg.norm(a - b, axis= 1)

# def manhattan_equation(a, b):
#     return np.sum(np.abs(a - b), axis=1)


def nearest_neighbors(train_data, p, k, distance):
    dist = distance(train_data, p)
    nearest_ind = np.argpartition(dist, k)[:k]
    # nearest_ind = nearest_ind[np.argsort(dist[nearest_ind])]
    return nearest_ind


def knn_prediction(train_data, labels, p, k, distance):
    nearest_ind = nearest_neighbors(train_data, p, k, distance)
    nearest_lab = labels[nearest_ind]
    return np.bincount(nearest_lab.astype(int)).argmax()

# first_person = binary_dev[0].reshape(1, -1)

#just prints out the indices and the nearest distances
# Euclidean
# nearest_ind_euclidean= nearest_neighbors(binary_train, first_person.flatten(), k=3, distance=euclidean_equation)
# print("Euclidean indices:", nearest_ind_euclidean)
# nearest_dist_eu = euclidean_equation(binary_train, first_person.flatten())
# print("Euclidean distances:", nearest_dist_eu[nearest_ind_euclidean])

# # Manhattan:
# nearest_ind_manhattan = nearest_neighbors(binary_train, first_person.flatten(), k=3, distance=manhattan_equation)
# print("Manhattan Indices:", nearest_ind_manhattan)
# nearest_dist_ma= manhattan_equation(binary_train, first_person.flatten())
# print("Manhattan Distances:", nearest_dist_ma[nearest_ind_manhattan])


mapping = {'<=50K': 0, '>50K': 1}
train_errs = []
dev_errs = []
best_dev_err = float('inf')
best_k = None

for k in range(1, 100, 2):
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(binary_train, target_train)
    
    # Predict on training set
    y_train_pred = knn.predict(binary_train)
    train_err = 1 - accuracy_score(target_train, y_train_pred)
    y_train_pred_num = np.array([mapping[label] for label in y_train_pred])
    train_pos_rate = np.mean(y_train_pred_num)
    
    # # Predict on dev set
    y_dev_pred = knn.predict(binary_dev)
    dev_err = 1 - accuracy_score(target_dev, y_dev_pred)
    y_dev_pred_num = np.array([mapping[label] for label in y_dev_pred])
    dev_pos_rate = np.mean(y_dev_pred_num)

    train_errs.append((train_err))
    dev_errs.append((dev_err))
    
    if dev_err < best_dev_err:
        best_dev_err = dev_err
        best_k = k

    print(f"k={k} train_err {train_err*100:.1f}% (+: {train_pos_rate:.1f}%) dev_err {dev_err*100:.1f}% (+: {dev_pos_rate*100:.1f}%)")


print(f"Best dev error rate: {best_dev_err*100:.1f}% achieved with k={best_k}")


prediction_test = pd.read_csv("income.test.blind.csv")
test_data = prediction_test[['age', 'sector','edu','marriage','occupation', 'race', 'sex', 'hours', 'country']]
binary_data_test = preprocessor.transform(test_data)
assert binary_train.shape[1] == binary_dev.shape[1] == binary_data_test.shape[1], "Mismatch in feature dimensions after encoding."
n_neighbors = 29
knn_final = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean", n_jobs=-1)
knn_final.fit(binary_train, target_train)
predictions2 = knn_final.predict(binary_data_test)
prediction_test['target'] = predictions2
prediction_test.to_csv('income.test.predicted.csv', index=False) 
prediction_test['target_num'] = prediction_test['target'].map(mapping)
test_pos_rate = prediction_test['target_num'].mean()
print(f"Positive rate in test set: {test_pos_rate*100:.2f}%")