# ridge implementation of part 2 
# citation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# # part 2 question 1
# data = pd.read_csv('my_train.csv')

# data = data.astype(str)

# # One Hot Encoding
# encoder = OneHotEncoder()
# binary_feats = encoder.fit_transform(data)

# # part 2 question 2
# for field, feats in data.items():
#     if field not in ['Id', 'SalePrice']:
#         unique_feats = len(np.unique(feats)) - 1
#     else:
#         unique_feats = len(np.unique(feats))
#     print(f"'{field}': '{unique_feats}'")

# new_total_feats = binary_feats.shape[1]
# print(f"Features after encoding: {new_total_feats}")

# part 2 question 3
my_train_data = pd.read_csv('my_train.csv')
my_dev_data = pd.read_csv('my_dev.csv')
my_test_data = pd.read_csv('test.csv')

my_train_data = my_train_data.astype(str).fillna('NA')
my_dev_data = my_dev_data.astype(str).fillna('NA')

test_data = my_test_data.astype(str).fillna('NA')

# One Hot Encoding
encoder = OneHotEncoder(handle_unknown='ignore')


# my train set 
x_mytrain = encoder.fit_transform(my_train_data.drop(columns=['Id','SalePrice']))
# x_mytrain = my_train_data['SalePrice']
y_mytrain = np.log1p(my_train_data['SalePrice'].astype(float))

# my dev set
x_mydev = encoder.transform(my_dev_data.drop(columns=['Id','SalePrice']))
# x_mydev = my_dev_data['SalePrice']
y_mydev = my_dev_data['SalePrice'].astype(float)

# ridge implementation
alpha_values = {'alpha': [0.1, 1.0, 10, 50, 100, 200]}
r_model = Ridge()
model = GridSearchCV(r_model, alpha_values, scoring='neg_mean_squared_error')
model.fit(x_mytrain, y_mytrain)

predictions = model.predict(x_mydev) 
predictions = np.expm1(predictions) # using expm1 instead of exp since after some research it seems to be more accurate

rmsle = np.sqrt(mean_squared_log_error(y_mydev, predictions))
print(f"RMSLE: {rmsle}")



# part 2 question 4
best_model = model.best_estimator_
top_10_positive = np.argsort(best_model.coef_)[-10:]
top_10_negative = np.argsort(best_model.coef_)[:10]

feat_names = encoder.get_feature_names_out()

top_10_positive_feats = [(feat_names[i], best_model.coef_[i]) for i in top_10_positive]
top_10_negative_feats = [(feat_names[i], best_model.coef_[i]) for i in top_10_negative]

# print("Top 10 positive")
# for feat, ind in reversed(top_10_positive_feats):
#     print(f"{feat}: {ind}")

# print("\nTop 10 negative")
# for feat, ind in top_10_negative_feats:
#     print(f"{feat}: {ind}")

# part 2 question 5
print(f'Bias: {best_model.intercept_}')
print(np.expm1(best_model.intercept_))

# dev error
dev_error = np.expm1(best_model.predict(x_mydev))
dev_error = np.sqrt(mean_squared_log_error(y_mydev, dev_error))
print(f'Dev Error: {dev_error}')

# part 2 question 7
prediction_test = pd.read_csv('test.csv')
prediction_test = prediction_test.astype(str).fillna('NA')
test_encoder = encoder.transform(prediction_test.drop(columns=['Id']))
test = np.expm1(best_model.predict(test_encoder))
dataframe = pd.DataFrame({'Id': prediction_test['Id'], 'SalePrice': test})
dataframe.to_csv('prediction.test4.1.csv', index=False)
