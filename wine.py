import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib

# dataset comes from this url:
# http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
dataset_url = 'data_import/winequality-red.csv'
data = pandas.read_csv(dataset_url, sep=';')
print(data.head())

# pandas setting to display a maximum of 20 columns
pandas.set_option('display.max_columns', 20)

data = pandas.read_csv(dataset_url, sep=';')
print(data.head())
print(data.shape)
print('-------------------------')
print(data.describe())

# split the data into training and test sets
y = data.quality
# drop the column named 'quality'
X = data.drop('quality', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=123,
                                                    stratify=y)

X_train_scaled = preprocessing.scale(X_train)
print('X_train_scaled')
print(X_train_scaled)
print('X_train_scaled_mean')
print(X_train_scaled.mean(axis=0))
print('X_train_scaled_std')
print(X_train_scaled.std(axis=0))

#fitting the Transformer API
scaler = preprocessing.StandardScaler().fit(X_train)
print(X_train_scaled.mean(axis=0))
print(X_train_scaled.std(axis=0))

X_test_scaled = scaler.transform(X_test)
print(X_test_scaled.mean(axis=0))
print(X_test_scaled.std(axis=0))

pipeline = make_pipeline(preprocessing.StandardScaler(),
                         RandomForestRegressor(n_estimators=100))

print(pipeline.get_params())

hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}

clf = GridSearchCV(pipeline, hyperparameters, cv=10)

# Fit and tune model
clf.fit(X_train, y_train)

print(clf.best_params_)

print(clf.refit)

y_pred = clf.predict(X_test)
print(r2_score(y_test, y_pred))

print(mean_squared_error(y_test, y_pred))

###########################################################################################
# # 3. Load red wine data.
# dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
# data = pd.read_csv(dataset_url, sep=';')
#
# # 4. Split data into training and test sets
# y = data.quality
# X = data.drop('quality', axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     test_size=0.2,
#                                                     random_state=123,
#                                                     stratify=y)
#
# # 5. Declare data preprocessing steps
# pipeline = make_pipeline(preprocessing.StandardScaler(),
#                          RandomForestRegressor(n_estimators=100))
#
# # 6. Declare hyperparameters to tune
# hyperparameters = {'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
#                    'randomforestregressor__max_depth': [None, 5, 3, 1]}
#
# # 7. Tune model using cross-validation pipeline
# clf = GridSearchCV(pipeline, hyperparameters, cv=10)
#
# clf.fit(X_train, y_train)
#
# # 8. Refit on the entire training set
# # No additional code needed if clf.refit == True (default is True)
#
# # 9. Evaluate model pipeline on test data
# pred = clf.predict(X_test)
# print(r2_score(y_test, pred))
# print(mean_squared_error(y_test, pred))