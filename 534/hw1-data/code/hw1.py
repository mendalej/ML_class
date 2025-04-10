import pandas as pd
import numpy as np

import sklearn.preprocessing

# # part 2
# # part 2 question 1
# data = pd.read_csv("toy.csv")

# print(data)

# data_age1 = data[['age']]
# print(data_age1)

# data_age2 = data['age']
# print(data_age2)

# data_subset = data[['age', 'sector']]
# print(data_subset)

# filtered_data = data[(data['age'] == 33) & (data['sector'] == 'Private')]
# print(filtered_data)

# data2 = data[['age','sector']]


# # part 2 question 2
# encoded_data = pd.get_dummies(data, columns=["age", "sector"])
# print(encoded_data)

# part 2 question 3
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

train_data = pd.read_csv("income.train.5k.csv")
dev_data = pd.read_csv("income.dev.csv")

# part 3 q  1

# num_processor = 'passthrough'
# cat_processor = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# part 3 q 2
num_processor = MinMaxScaler(feature_range=(0, 2))
cat_processor = OneHotEncoder(sparse_output=False, handle_unknown='ignore')


# part 3 Q 1
preprocessor = ColumnTransformer([
    ('num', num_processor, ['age', 'hours']),
    ('cat', cat_processor, ['sector', 'edu', 'marriage', 'occupation', 'race', 'sex', 'country'])
])


preprocessor.fit(train_data)
binary_data = preprocessor.transform(train_data)
binary_dev = preprocessor.transform(dev_data)


# part 2 Q 4

# data_train = train_data[['age', 'sector', 'edu', 'marriage', 'occupation', 'race', 'sex', 'hours', 'country']]
# data_dev = dev_data[['age', 'sector', 'edu', 'marriage', 'occupation', 'race', 'sex', 'hours', 'country']]

# encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# encoder_dev = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# encoder.fit(data_train) 
# # encoder_dev.fit(data_dev)

# binary_data = encoder.transform(data_train)
# binary_dev = encoder.transform(data_dev)

target_train = train_data['target']
target_dev = dev_data['target']

# print(binary_data)


# part 3 Q 1
feature = preprocessor.get_feature_names_out()
feature2 = len(feature)
print(f"Feature dimension: {feature2}")

# part 2 question 4
# feature = encoder.get_feature_names_out()
# feature2 = len(feature)
# print(f"Feature dimension: {feature2}")
# print(feature2)



# part 2 question 4

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

mapping = {'<=50K': 0, '>50K': 1}
train_errs = []
dev_errs = []
best_dev_err = float('inf')
best_k = None

for k in range(1, 100, 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(binary_data, target_train)
    
    # Predict on training set
    y_train_pred = knn.predict(binary_data)
    train_err = 1 - accuracy_score(target_train, y_train_pred)
    y_train_pred_num = [mapping[label] for label in y_train_pred]
    train_pos_rate = sum(y_train_pred_num) / len(y_train_pred_num)
    # y_train_pred_num = np.array([mapping[label] for label in y_train_pred])
    # train_pos_rate = np.mean(y_train_pred_num)
    
    # # Predict on dev set
    y_dev_pred = knn.predict(binary_dev)
    dev_err = 1 - accuracy_score(target_dev, y_dev_pred)
    y_dev_pred_num = [mapping[label] for label in y_dev_pred]
    dev_pos_rate = sum(y_dev_pred_num) / len(y_dev_pred_num)
    # y_dev_pred_num = np.array([mapping[label] for label in y_dev_pred])
    # dev_pos_rate = np.mean(y_dev_pred_num)

    
    train_errs.append((train_err))
    dev_errs.append((dev_err))
    
    if dev_err < best_dev_err:
        best_dev_err = dev_err
        best_k = k

    print(f"k={k} train_err {train_err*100:.1f}% (+: {train_pos_rate:.1f}%) dev_err {dev_err*100:.1f}% (+: {dev_pos_rate*100:.1f}%)")


    # print(f"k={k} train_err={train_err*100:.1f} dev_err={dev_err:.4f}, train_pos_rate={train_pos_rate:.4f}")

# for k, train_error, train_positive_rate in train_errs:
#     dev_error, dev_positive_rate = next((d[1], d[2]) for d in dev_errs if d[0] == k)
#     print(f"k={k} train_err {train_error*100:.1f}% (+: {train_positive_rate*100:.1f}%) dev_err {dev_error*100:.1f}% (+: {dev_positive_rate*100:.1f}%)")

print(f"Best dev error rate: {best_dev_err*100:.1f}% achieved with k={best_k}")

knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(binary_data, target_train)
y_test_pred = knn_best.predict(binary_dev)


prediction_test = pd.read_csv("income.test.blind.csv")
test_data = prediction_test[['age', 'sector','edu','marriage','occupation', 'race', 'sex', 'hours', 'country']]
test_encoder = OneHotEncoder(sparse_output = False, handle_unknown='ignore')
test_encoder.fit(test_data)
binary_data_test = preprocessor.transform(test_data)
n_neighbors = 41
predictions2 = knn.predict(binary_data_test)
 
prediction_test['target'] = predictions2
 
prediction_test.to_csv('income_predictions.csv', index=False) # goal is .185 but I got .190