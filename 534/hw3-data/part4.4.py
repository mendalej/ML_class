# this is my original version of part 3 
# citations: https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/
# citations: https://xgboost.readthedocs.io/en/stable/python/python_intro.html
# citations: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
# citations: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
# citations: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
# citations: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
# citations: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
# citations: https://www.kaggle.com/code/wwwittt/blend-model

import pandas as pd
import numpy as np  
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

 
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
x_train[num_cats] = x_train[num_cats]
x_dev[cat_cats] = x_dev[cat_cats].astype(str).fillna(0)
x_dev[num_cats] = x_dev[num_cats]
 
 
# part 3 q 2
# uses code from hw1
 
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), num_cats),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), cat_cats)
    ]
)


preprocessor.fit(my_train_data.drop(columns=['Id', 'SalePrice']))
x_mytrain = preprocessor.transform(x_train)
y_mytrain = np.log1p(my_train_data['SalePrice'].astype(float))
 
# my dev set
x_mydev = preprocessor.transform(x_dev)
# x_mydev = my_dev_data['SalePrice']
y_mydev = my_dev_data['SalePrice'].astype(float)
 

# model = ElasticNet(alpha= .00001, l1_ratio=1) #BEST SO FAR
# model = Lasso(alpha= .0001)
# model = SVR(kernel='rbf', C=1, epsilon=0.01)  #BEST SO FAR @ .121
# model = GradientBoostingRegressor(n_estimators=10000, learning_rate=0.1, max_depth=1) 
# model = RandomForestRegressor(n_estimators=100, max_depth=1000, random_state=0) 
# model.fit(x_mytrain, y_mytrain)
# used inspritation from kaggle citation above (mostly just to know what to add to the param_grid)
param_grid = {
    'n_estimators': [3460],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [1, 3, 5], 
    'gamma' : [0, 0.1, 0.2],
    'subsample': [0.5, 1],
    'colsample_bytree': [0.5, 0.7,  1],
    'reg_alpha': [0.00006, 0.1, 1],
    'objective': ['reg:squarederror'],
    'seed': [27],
    'random_state': [42],
    'min_child_weight': [0, 1, 3, 5]
}

model = xgb.XGBRegressor() # best so far at .120 at 10000
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_log_error', verbose=1, n_jobs=-1)
grid.fit(x_mytrain, y_mytrain)
best = grid.best_params_
 

best_model = xgb.XGBRegressor(**best)
best_model.fit(x_mytrain, y_mytrain)


predictions = best_model.predict(x_mydev)
predictions = np.expm1(predictions) # using expm1 instead of exp since after some research it seems to be more accurate


rmsle = np.sqrt(mean_squared_log_error(y_mydev, predictions))
print(f"RMSLE: {rmsle}")
 

prediction_test = pd.read_csv('test.csv')
prediction_test = prediction_test.astype(str).fillna('NA')
test_encoder = preprocessor.transform(prediction_test.drop(columns=['Id']))
test = np.expm1(best_model.predict(test_encoder))
dataframe = pd.DataFrame({'Id': prediction_test['Id'], 'SalePrice': test})
dataframe.to_csv('prediction.test4.4.csv', index=False)