import numpy as np
import pandas as pd
import math as ma

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor

################################################################
#### DATA DATA DATA  
################################################################
def loadIris():
	df = pd.read_csv("./datasets/iris_binned.csv", header=None)
	# print(df.head())
	X = df.drop(columns=[28])
	y = df[[28]]
	return X, y

def loadVoting():
	df = pd.read_csv("./datasets/house_voting_84_post.csv", header=None)
	# print(df.head())
	X = df.drop(columns=[16])
	y = df[[16]]

	# label encode y
	le = LabelEncoder()
	le.fit(y.values.ravel())
	y = le.transform(y.values.ravel())
	# return
	return X, y

def loadAuto():
	df = pd.read_csv("./datasets/auto_mpg_post.csv", header=None)
	# print(df.head())
	X = df.drop(columns=[7])
	y = df[[7]]
	return X, y


################################################################
#### Iris-Binned 
################################################################
X, y = loadIris()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=52)

# (1) KNN Classifier
best_cross_val_score = 0
best_k = 1
for k in range(1,15):
	classifier = KNeighborsClassifier(n_neighbors=k)
	model = classifier.fit(X_train, y_train.values.ravel())
	predictions = model.predict(X_test)

	metric = accuracy_score(y_test, predictions)

	print("CV ACCURACY for (Iris-Binned) n_neighbors={}: {}".format(k, metric))
	if metric >= best_cross_val_score:
		best_cross_val_score = metric
		best_k = k

# print(" ")
print("--> BEST CV ACCURACY for (Iris-Binned) n_neighbors={}: {}".format(best_k, best_cross_val_score))
print(" ")

# (2) Decision Tree Classifier

classifier = DecisionTreeClassifier(random_state=42)
model = classifier.fit(X_train, y_train)
predictions = model.predict(X_test)

metric = accuracy_score(y_test, predictions)
print("--> BEST TEST ACCURACY for DTree (Iris): {}".format(metric))
print(" ")

# (3) Naive Bayes

clf = GaussianNB()
clf.fit(X_train, y_train.values.ravel())
predictions = clf.predict(X_test)
metric = accuracy_score(y_test, predictions)
print("--> BEST TEST ACCURACY for NBayes (Iris): {}".format(metric))
print(" ")

clf_pf = GaussianNB()
clf_pf.partial_fit(X_train, y_train.values.ravel(), np.unique(y_train))
predictions = clf_pf.predict(X_test)
metric = accuracy_score(y_test, predictions)
print("--> BEST TEST ACCURACY for NBayes-Partial-Fit (Iris): {}".format(metric))
print(" ")

# (A) Bagging

best_cross_val_score = 0
best_k = 100
for k in range(100, 1000, 100):
	classifier = BaggingClassifier(n_estimators=k)
	scores = cross_val_score(classifier, X_train, y_train.values.ravel(), cv=5, scoring=make_scorer(accuracy_score))
	metric = scores.mean()

	print("CV ACCURACY for (Iris_Binned) n_estimators={}: {}".format(k, metric))
	if metric > best_cross_val_score:
		best_cross_val_score = metric
		best_k = k

# print(" ")
print("--> BEST CV ACCURACY for Bagging n_estimators={} (Iris_Binned): {}".format(best_k, metric))
print(" ")

# (B) AdaBoost

best_cross_val_score = 0
best_k = 100
for k in range(100, 600, 100):
	classifier = AdaBoostClassifier(n_estimators=k)
	scores = cross_val_score(classifier, X_train, y_train.values.ravel(), cv=5)
	metric = scores.mean()

	print("CV ACCURACY for (Iris-Binned) n_estimators={}: {}".format(k, metric))
	if metric > best_cross_val_score:
		best_cross_val_score = metric
		best_k = k

# print(" ")
print("--> BEST CV ACCURACY for AdaBoost n_estim..={} (Iris): {}".format(best_k, metric))
print(" ")


# (C) RandomForest

best_cross_val_score = 0
best_k = 100
for k in range(100, 1000, 100):
	classifier = RandomForestClassifier(n_estimators=k)
	scores = cross_val_score(classifier, X_train, y_train.values.ravel(), cv=5, scoring=make_scorer(accuracy_score))
	metric = scores.mean()

	print("CV ACCURACY for (Iris_Binned) n_estimators={}: {}".format(k, metric))
	if metric > best_cross_val_score:
		best_cross_val_score = metric
		best_k = k

# print(" ")
print("--> BEST CV ACCURACY for Random Forest n_estimators={} (Iris_Binned): {}".format(best_k, metric))
print(" ")



################################################################
#### Voting 
################################################################
X, y = loadVoting()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=52)

# (1) KNN Classifier
best_cross_val_score = 0
best_k = 1
for k in range(1,15):
	classifier = KNeighborsClassifier(n_neighbors=k)
	model = classifier.fit(X_train, y_train)
	predictions = model.predict(X_test)

	metric = accuracy_score(y_test, predictions)

	print("CV ACCURACY for (Voting) n_neighbors={}: {}".format(k, metric))
	if metric > best_cross_val_score:
		best_cross_val_score = metric
		best_k = k

# print(" ")
print("--> BEST CV ACCURACY for (Voting) n_neighbors={}: {}".format(best_k, best_cross_val_score))
print(" ")

# (2) Decision Tree Classifier

classifier = DecisionTreeClassifier(random_state=42)
model = classifier.fit(X_train, y_train)
predictions = model.predict(X_test)

metric = accuracy_score(y_test, predictions)
print("--> BEST TEST ACCURACY for DTree (Voting): {}".format(metric))
print(" ")

# (3) Naive Bayes

clf = GaussianNB()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
metric = accuracy_score(y_test, predictions)
print("--> BEST TEST ACCURACY for NBayes (Voting): {}".format(metric))
print(" ")

clf_pf = GaussianNB()
clf_pf.partial_fit(X_train, y_train, np.unique(y_train))
predictions = clf_pf.predict(X_test)
metric = accuracy_score(y_test, predictions)
print("--> BEST TEST ACCURACY for NBayes-Partial-Fit (Voting): {}".format(metric))
print(" ")



# (A) Bagging

best_cross_val_score = 0
best_k = 100
for k in range(100, 1000, 100):
	classifier = BaggingClassifier(n_estimators=k)
	scores = cross_val_score(classifier, X_train, y_train, cv=5, scoring=make_scorer(accuracy_score))
	metric = scores.mean()

	print("CV ACCURACY for (Voting) n_estimators={}: {}".format(k, metric))
	if metric > best_cross_val_score:
		best_cross_val_score = metric
		best_k = k

# print(" ")
print("--> BEST CV ACCURACY for Bagging n_estimators={} (Voting): {}".format(best_k, metric))
print(" ")


# (B) AdaBoost

best_cross_val_score = 0
best_k = 100
for k in range(100, 600, 100):
	classifier = AdaBoostClassifier(n_estimators=k)
	scores = cross_val_score(classifier, X_train, y_train, cv=5)
	metric = scores.mean()

	print("CV ACCURACY for (Voting) n_estimators={}: {}".format(k, metric))
	if metric > best_cross_val_score:
		best_cross_val_score = metric
		best_k = k

# print(" ")
print("--> BEST CV ACCURACY for AdaBoost n_estim..={} (Voting): {}".format(best_k, metric))
print(" ")



# (C) RandomForest

best_cross_val_score = 0
best_k = 100
for k in range(100, 1000, 100):
	classifier = RandomForestClassifier(n_estimators=k)
	scores = cross_val_score(classifier, X_train, y_train, cv=5, scoring=make_scorer(accuracy_score))
	metric = scores.mean()

	print("CV ACCURACY for (Voting) n_estimators={}: {}".format(k, metric))
	if metric > best_cross_val_score:
		best_cross_val_score = metric
		best_k = k

# print(" ")
print("--> BEST CV ACCURACY for Random Forest n_estimators={} (Voting): {}".format(best_k, metric))
print(" ")




################################################################
#### Auto-Mpg 
################################################################
X, y = loadAuto()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=52)

# (1) KNN 

best_cross_val_score = 10000
best_k = 1
for k in range(1,15):
	regressor = KNeighborsRegressor(n_neighbors=k)
	model = regressor.fit(X_train, y_train)
	predictions = model.predict(X_test)

	metric = mean_absolute_error(y_test, predictions)

	print("CV ACCURACY for (Auto-Mpg) n_neighbors={}: {}".format(k, metric))
	if metric <= best_cross_val_score:
		best_cross_val_score = metric
		best_k = k

# print(" ")
print("--> BEST TEST ACCURACY for (Auto-Mpg) n_neighbors={}: {}".format(best_k, best_cross_val_score))
print(" ")


# (2) Decision Tree Classifier

classifier = DecisionTreeRegressor(random_state=42)
model = classifier.fit(X_train, y_train)
predictions = model.predict(X_test)

metric = mean_absolute_error(y_test, predictions)
print("--> BEST TEST ACCURACY for DTree (Auto-Mpg): {}".format(metric))
print(" ")

# (3) NN / MLP

regressor = BayesianRidge()
model = regressor.fit(X_train, y_train.values.ravel())
predictions = model.predict(X_test)

metric = mean_absolute_error(y_test, predictions)
print("--> BEST TEST ACCURACY for Naive Bayesian Ridge (Auto-Mpg): {}".format(metric))
print(" ")



# (A) Bagging

best_cross_val_score = 10000
best_k = 0.1
for k in np.arange(0.1, 1.0, 0.1):
	regressor = BaggingRegressor(max_samples=k)
	scores = cross_val_score(regressor, X_train, y_train.values.ravel(), cv=5, scoring=make_scorer(mean_absolute_error))
	metric = scores.mean()

	print("CV ACCURACY for (Auto-MPG) max_samples={}: {}".format(round(k,2), metric))
	if metric < best_cross_val_score:
		best_cross_val_score = metric
		best_k = k

# print(" ")
print("--> BEST CV ACCURACY for Bagging max_samples={} (Auto-MPG): {}".format(best_k, metric))
print(" ")

# (B) AdaBoost

best_cross_val_score = 10000
best_k = 100
for k in range(100, 1000, 100):
	regressor = AdaBoostRegressor(n_estimators=k)
	scores = cross_val_score(regressor, X_train, y_train.values.ravel(), cv=5, scoring=make_scorer(mean_absolute_error))
	metric = scores.mean()

	print("CV ACCURACY for (Auto-MPG) n_estimators={}: {}".format(k, metric))
	if metric < best_cross_val_score:
		best_cross_val_score = metric
		best_k = k

# print(" ")
print("--> BEST CV ACCURACY for AdaBoost n_estim..={} (Auto-MPG): {}".format(best_k, metric))
print(" ")

# (C) RandomForest

best_cross_val_score = 10000
best_k = 100
for k in range(100, 1000, 100):
	regressor = RandomForestRegressor(n_estimators=k)
	scores = cross_val_score(regressor, X_train, y_train.values.ravel(), cv=5, scoring=make_scorer(mean_absolute_error))
	metric = scores.mean()

	print("CV ACCURACY for (Auto-MPG) n_estimators={}: {}".format(k, metric))
	if metric < best_cross_val_score:
		best_cross_val_score = metric
		best_k = k

# print(" ")
print("--> BEST CV ACCURACY for Random Forest n_estimators={} (Auto-MPG): {}".format(best_k, metric))
print(" ")



