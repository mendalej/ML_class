# this is version 2 of part3.py 
# it got a lower kaggle score than part3.py but has a higher rsmle score
# this code does not include the top 10 code but it is in the original version of part3.py

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
num_processor = SimpleImputer(strategy='mean')
cat_processor = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
 
preprocessor = ColumnTransformer([
    ('num', num_processor, num_cats),
    ('cat', cat_processor, cat_cats)
])

preprocessor.fit(x_train, y_train)
x_mytrain = preprocessor.transform(x_train)
x_mydev = preprocessor.transform(x_dev)
# y_mytrain = np.log1p(my_train_data['SalePrice'].astype(float))
y_mytrain = np.log1p(y_train.astype(float))

new_total_feats = x_mytrain.shape[1]
print(f"Features after encoding: {new_total_feats}")

model = LinearRegression()
model.fit(x_mytrain, y_mytrain)

predictions = model.predict(x_mydev) 
predictions = np.expm1(predictions) # using expm1 instead of exp since after some research it seems to be more accurate

rmsle = np.sqrt(mean_squared_log_error(y_dev, predictions))
print(f"RMSLE: {rmsle}")

# part 2 question 7
prediction_test = pd.read_csv('test.csv')
prediction_test = prediction_test.astype(str).fillna('NA')
test_encoder = preprocessor.transform(prediction_test.drop(columns=['Id']))
test = np.expm1(model.predict(test_encoder))
dataframe = pd.DataFrame({'Id': prediction_test['Id'], 'SalePrice': test})
dataframe.to_csv('prediction.test.csv', index=False)
