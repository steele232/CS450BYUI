import numpy as np
import pandas as pd
import math as ma




################################################################
#### IRIS @ DEFAULT, AS-IS @ SKLearn Decision Tree
################################################################

# *********** LOAD IRIS DATA ***********
from sklearn import datasets
iris = datasets.load_iris()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=52)

# *********** @IRIS @SKLearn Decision Tree ***********
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier()
model = clf.fit(X_train, y_train)
predictions = model.predict(X_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, predictions)
print(" ")
print("ACCURACY ==> IRIS @ DEFAULT, AS-IS @ SKLearn MLP CLASSIFIER : {}".format(acc))
print(" ")







################################################################
#### IRIS @ BINNING @ SKLearn Decision Tree
################################################################
iris = datasets.load_iris()

# *********** DISCRETIZE IRIS DATA ***********
from sklearn.preprocessing import KBinsDiscretizer

# Try binning with a few different values for K Bins
for i in range(2,10):
    enc = KBinsDiscretizer(n_bins=i, encode='onehot')
    X_binned = enc.fit_transform(iris.data)

    X_train, X_test, y_train, y_test = train_test_split(X_binned, iris.target, test_size=0.33, random_state=52)


    for maxiter in range(200, 500, 50):

        clf = MLPClassifier(max_iter=maxiter)
        model = clf.fit(X_train, y_train)
        predictions = model.predict(X_test)

        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test, predictions)
        print("ACCURACY ==> IRIS @ BINNING with " + str(i) + " bins @ SKLearn MLP CLASSIFIER  w/ max_iter={}: {}".format(maxiter, acc))


print(" ")



################################################################
#### AUTO-MPG @ SKLearn Decision Tree Regressor
################################################################
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


dataset = pd.read_csv("./auto-mpg.csv", header=None, 
                        delim_whitespace=True, na_values="?")
dataset.dropna(axis=0, inplace=True, how='any') # dropping rows
X = dataset.drop(columns=[0,8]) # take out targets, 
                                # and column with unique strings
y = dataset[[0]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=52)

adamResults = []
for i in range(2, 50):
    classifier = MLPRegressor(random_state=42, hidden_layer_sizes=(i,), solver='adam')
    model = classifier.fit(X_train, y_train.values.ravel())
    predictions = model.predict(X_test)

    metric = mean_absolute_error(y_test, predictions)
    # print(" ")
    print("ACCURACY ==> AUTO-MPG @ SKLearn MLP Regressor w/ solver=adam @ {} Hidden Layer Nodes: {}".format(i, metric))
    # print("I believe this means that the regression model we made ")
    # print("is, on average, within +/- {} mpg of the actual ".format(round(metric,2)))
    # print(" ")
    adamResults.append([i,metric])

print(" ")

lfbgsResults = []
for i in range(2, 50):
    classifier = MLPRegressor(random_state=42, hidden_layer_sizes=(i,), solver='lbfgs')
    model = classifier.fit(X_train, y_train.values.ravel())
    predictions = model.predict(X_test)

    metric = mean_absolute_error(y_test, predictions)
    # print(" ")
    print("ACCURACY ==> AUTO-MPG @ SKLearn MLP Regressor w/ solver=lbfgs @ {} Hidden Layer Nodes: {}".format(i, metric))
    # print("I believe this means that the regression model we made ")
    # print("is, on average, within +/- {} mpg of the actual ".format(round(metric,2)))
    # print(" ")
    lfbgsResults.append([i,metric])


print(" ")
# accrue best score w/ adam
minResultAdam = 100000
minResultAdamNumNodes = 0
for result in adamResults:
    if result[1] < minResultAdam:
        minResultAdam = result[1]
        minResultAdamNumNodes = result[0]
print("Best Adam score: {} @ Using {} Nodes".format(minResultAdam, minResultAdamNumNodes))

# accrue best score w/ lbfgs
minResultLfbgs = 100000 
minResultLfbgsNumNodes= 0
for result in lfbgsResults:
    if result[1] < minResultLfbgs:
        minResultLfbgs = result[1]
        minResultLfbgsNumNodes = result[0]
print("Best Lfbgs score: {} @ Using {} Nodes".format(minResultLfbgs, minResultLfbgsNumNodes))









print(" ")

