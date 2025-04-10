# this is adapted from the second version of the code from part 3
# I am editing it for non linear regression I used method 3
# citations: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html

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
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures



# reading data
my_train_data = pd.read_csv('my_train.csv')
my_dev_data = pd.read_csv('my_dev.csv')
mytest_data = pd.read_csv('test.csv')

# my_train_data = my_train_data.astype(str).fillna(0)
# my_dev_data = my_dev_data.astype(str).fillna(0)

x_dev = my_dev_data.drop(columns=['Id','SalePrice'])
y_dev = my_dev_data['SalePrice']


x_train = my_train_data.drop(columns=['Id','SalePrice'])
y_train = my_train_data['SalePrice']

test = mytest_data.drop(columns=['Id'])

cat_cats = x_train.select_dtypes(include=['object']).columns
num_cats = x_train.select_dtypes(exclude=['object']).columns

important_features = ['OverallArea', 'LotArea'] 
other_features = [col for col in num_cats if col not in important_features]

mytest_data[cat_cats] = mytest_data[cat_cats].astype(str).fillna('NA')
mytest_data[num_cats] = mytest_data[num_cats].astype(float).fillna(0)


important_features = ['OverallArea', 'LotArea'] 
important_features = [item for item in important_features if item in x_train.columns]


x_train[cat_cats] = x_train[cat_cats].astype(str).fillna('NA')
x_train[num_cats] = x_train[num_cats].astype(float).fillna(0)
x_dev[cat_cats] = x_dev[cat_cats].astype(str).fillna('NA')
x_dev[num_cats] = x_dev[num_cats].astype(float).fillna(0)


# part 3 q 2
# uses code from hw1
# num_processor = SimpleImputer(strategy='mean')
# cat_processor = OneHotEncoder(handle_unknown='ignore', sparse_output=True)

processor = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

num_processor = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_processor = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
 
 
preprocessor = ColumnTransformer([
    ('key', processor, important_features),
    ('num', num_processor, other_features),
    ('cat', cat_processor, cat_cats)
])

model = make_pipeline(
    preprocessor,
    # I am using 1 instead of 2 bc I get a much better result
    PolynomialFeatures(degree=1, include_bias=False),
    LinearRegression()
)


model.fit(x_train, np.log1p(y_train))


predictions = model.predict(x_dev)
predictions = np.expm1(predictions) # using expm1 instead of exp since after some research it seems to be more accurate
predictions = np.maximum(predictions, 0)


rmsle = np.sqrt(mean_squared_log_error(y_dev, predictions))
print(f"RMSLE: {rmsle}")


prediction_test = pd.read_csv('test.csv')
prediction_test[cat_cats] = prediction_test[cat_cats].astype(str).fillna('NA')
prediction_test[num_cats] = prediction_test[num_cats].astype(float).fillna(0)
test_encoder = model.predict(prediction_test.drop(columns=['Id']))
test_encoder = np.expm1(test_encoder)
test = np.maximum(test_encoder, 0)
dataframe = pd.DataFrame({'Id': prediction_test['Id'], 'SalePrice': test})
dataframe.to_csv('prediction.test4.3.csv', index=False)
