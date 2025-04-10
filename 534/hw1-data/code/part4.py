import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
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

preprocessor.fit(train_data)
binary_train = preprocessor.transform(train_data)
binary_dev = preprocessor.transform(dev_data) 



# following code is added so that the visuals are the same as provided in the example

# feat_num = ['age', 'hours']
# feat_cat = preprocessor.named_transformers_['cat'].get_feature_names_out(['sector', 'edu', 'marriage', 'occupation', 'race', 'sex', 'country'])

# no_count = [f for f in feat_cat if not f.startswith('country')]
# count = [f for f in feat_cat if f.startswith('country')]

# feats = ['age'] + no_count + ['hours'] + count

# binary_train_df = pd.DataFrame(binary_train, columns=feat_num + list(feat_cat))[feats]
# binary_dev_df = pd.DataFrame(binary_dev, columns=feat_num + list(feat_cat))[feats]

# print(binary_dev_df.head())

# code copied from part 3 with some extra things added to calculate the proper distances
first_person = binary_dev[0].reshape(1, -1)
# euclidean code


knn_e = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
knn_e.fit(binary_train, train_data['target'])
dist_e, ind_e = knn_e.kneighbors(first_person)


print("Indices of the three closest individuals:", ind_e[0])
print("Distances to the three closest individuals:", dist_e[0])

# manhattan code

knn_m = KNeighborsClassifier(n_neighbors=3, metric="manhattan")
knn_m.fit(binary_train, train_data['target'])
dist_m, ind_m = knn_m.kneighbors(first_person)

print("Indices of the three closest individuals:", ind_m[0])
print("Distances to the three closest individuals:", dist_m[0])

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
n_neighbors = 37
knn_final = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean", n_jobs=-1)
knn_final.fit(binary_train, target_train)
predictions2 = knn_final.predict(binary_data_test)
prediction_test['target'] = predictions2
prediction_test.to_csv('income.test.predicted.csv', index=False) 
prediction_test['target_num'] = prediction_test['target'].map(mapping)
test_pos_rate = prediction_test['target_num'].mean()
print(f"Positive rate in test set: {test_pos_rate*100:.2f}%")
