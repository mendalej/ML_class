# this is the code without the print statements and without the commented out code
# and without the code needed for the data to be produced
# this is exactly the same as the code in part4.py
# this an extra file
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# code copied from part 3
train_data = pd.read_csv('income.train.5k.csv')
dev_data = pd.read_csv('income.dev.csv')

train_feats = train_data.drop(columns=['target'])
dev_feats = dev_data.drop(columns=['target'])

num_processor = MinMaxScaler(feature_range=(0, 2))
cat_processor = OneHotEncoder(sparse_output=False, handle_unknown='ignore')


preprocessor = ColumnTransformer([
    ('num', num_processor, ['age', 'hours']),
    ('cat', cat_processor, ['sector', 'edu', 'marriage', 'occupation', 'race', 'sex', 'country'])
])

preprocessor.fit(train_data)
binary_train = preprocessor.transform(train_data)
binary_dev = preprocessor.transform(dev_data) 

# following code is added so that the visuals are the same as provided in the example

feat_num = ['age', 'hours']
feat_cat = preprocessor.named_transformers_['cat'].get_feature_names_out(['sector', 'edu', 'marriage', 'occupation', 'race', 'sex', 'country'])

no_count = [f for f in feat_cat if not f.startswith('country')]
count = [f for f in feat_cat if f.startswith('country')]

feats = ['age'] + no_count + ['hours'] + count

binary_train_df = pd.DataFrame(binary_train, columns=feat_num + list(feat_cat))
binary_train_df = binary_train_df[feats]

binary_dev_df = pd.DataFrame(binary_dev, columns=feat_num + list(feat_cat))
binary_dev_df = binary_dev_df[feats]

print(binary_dev_df.head())

# code copied from part 3 with some extra things added to calculate the proper distances

# euclidean code

knn_e = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
knn_e.fit(binary_train, train_data['target'])

first_person = binary_dev[0].reshape(1, -1)

dist_e, ind_e = knn_e.kneighbors(first_person)


print("Indices of the three closest individuals:", ind_e[0])
print("Distances to the three closest individuals:", dist_e[0])

# manhattan code

knn_m = KNeighborsClassifier(n_neighbors=3, metric="manhattan")

knn_m.fit(binary_train, train_data['target'])

dist_m, ind_m = knn_m.kneighbors(first_person)

print("Indices of the three closest individuals:", ind_m[0])
print("Distances to the three closest individuals:", dist_m[0])


# numpy implementation
