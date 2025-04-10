# this uses the code of part3v2.py
# the ridge implementation of part 3
# citation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# reading data
my_train_data = pd.read_csv('my_train.csv')
my_dev_data = pd.read_csv('my_dev.csv')
my_test_data = pd.read_csv('test.csv')

# my_train_data = my_train_data.astype(str).fillna(0)
# my_dev_data = my_dev_data.astype(str).fillna(0)

x_dev = my_dev_data.drop(columns=['Id','SalePrice'])
y_dev = my_dev_data['SalePrice']

x_train = my_train_data.drop(columns=['Id','SalePrice'])
y_train = my_train_data['SalePrice']

test = my_test_data.drop(columns=['Id'])

cat_cats = x_train.select_dtypes(include=['object']).columns
num_cats = x_train.select_dtypes(exclude=['object']).columns

x_train[cat_cats] = x_train[cat_cats].astype(str).fillna(0)
x_train[num_cats] = x_train[num_cats].astype(float)
x_dev[cat_cats] = x_dev[cat_cats].astype(str).fillna(0)
x_dev[num_cats] = x_dev[num_cats].astype(float)


# part 3 q 2
# uses code from hw1
num_processor = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())  
])
cat_processor = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
 
preprocessor = ColumnTransformer([
    ('num', num_processor, num_cats),
    ('cat', cat_processor, cat_cats)
])

preprocessor.fit(x_train, y_train)
x_mytrain = preprocessor.transform(x_train)
x_mydev = preprocessor.transform(x_dev)
# y_mytrain = np.log1p(my_train_data['SalePrice'].astype(float))
y_mytrain = np.log1p(y_train.astype(float))
y_mydev = np.log1p(my_dev_data['SalePrice'].astype(float))


# my dev set
# x_mydev = preprocessor.transform(x_dev)
# # x_mydev = my_dev_data['SalePrice']
# y_mydev = my_dev_data['SalePrice'].astype(float)

# part 3 question 3(a)
# for field, feats in my_train_data.items():
#     if field not in ['Id', 'SalePrice']:
#         unique_feats = len(np.unique(feats.astype(str))) - 1
#     else:
#         unique_feats = len(np.unique(feats))
#     print(f"'{field}': '{unique_feats}'")

new_total_feats = x_mytrain.shape[1]
print(f"Features after encoding: {new_total_feats}")

alpha_values = {'alpha': [0.1, 0.01, 0.001, 0.0001, 0.00001, 1, 10, 100, 1000, 10000, 100000]}
r_model = Ridge()
model = GridSearchCV(r_model, alpha_values, scoring='neg_mean_squared_error')
model.fit(x_mytrain, y_mytrain)

# model = LinearRegression()
# for part 4 q 1
# model = Ridge(alpha = .09)
# model.fit(x_mytrain, y_mytrain)

predictions = model.predict(x_mydev) 
predictions = np.expm1(predictions) # using expm1 instead of exp since after some research it seems to be more accurate

rmsle = np.sqrt(mean_squared_log_error(y_dev, predictions))
print(f"RMSLE: {rmsle}")

# part 3 question 3(c)
top_10_positive = np.argsort(model.best_estimator_.coef_)[-10:]
top_10_negative = np.argsort(model.best_estimator_.coef_)[:10]

feat_names = preprocessor.get_feature_names_out()

top_10_positive_feats = [(feat_names[i], model.best_estimator_.coef_[i]) for i in top_10_positive]
top_10_negative_feats = [(feat_names[i], model.best_estimator_.coef_[i]) for i in top_10_negative]

# print("Top 10 positive")
# for feat, ind in reversed(top_10_positive_feats):
#     print(f"{feat}: {ind}")

# print("\nTop 10 negative")
# for feat, ind in top_10_negative_feats:
#     print(f"{feat}: {ind}")

# part 2 question 5
print(f'Bias: {model.best_estimator_.intercept_}')
print(np.expm1(model.best_estimator_.intercept_))

# part 2 question 7
prediction_test = pd.read_csv('test.csv')
prediction_test[cat_cats] = prediction_test[cat_cats].astype(str).fillna('NA')
prediction_test[num_cats] = prediction_test[num_cats].astype(float)
test_encoder = preprocessor.transform(prediction_test.drop(columns=['Id']))
test = np.expm1(model.predict(test_encoder))
dataframe = pd.DataFrame({'Id': prediction_test['Id'], 'SalePrice': test})
dataframe.to_csv('prediction.test4.2.csv', index=False)
