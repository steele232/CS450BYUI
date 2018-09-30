

# *********** LOAD DATA ***********
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()

# Show the data (the attributes of each instance)
# print(iris.data)

# Show the target values (in numeric format) of each instance
# print(iris.target)

# Show the actual target names that correspond to each number
# print(iris.target_names)

# *********** PREPARE DATA ***********
# used code from example in docs : 
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=42)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

# *********** USE AN EXISTING ALGORITHM TO CREATE A MODEL ***********
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
model = classifier.fit(X_train, y_train)


# *********** USE THAT MODEL TO MAKE PREDICTIONS ***********
y_predicted = model.predict(X_test)
# now compare results.
sumCorrect = 0
sumError = 0
for i in range(0,len(y_predicted)):
	if y_predicted[i] == y_test[i]:
		sumCorrect += 1
	else:
		sumError += 1

percentError = 100 * (sumError / (sumError + sumCorrect))

print("Percent Error: {} %".format(percentError))
print("Sum Error: {}".format(sumError))
print("Sum Correct: {}".format(sumCorrect))


# *********** IMPLEMENT YOUR OWN NEW "ALGORITHM" ***********

class HardCodedClassifier:
	def fit(self, X_train, y_train):
		model = HardCodedModel()
		return model

class HardCodedModel:
	def predict(self, data_test):
		length = len(data_test)
		predictions = np.zeros(length)
		return predictions

classifier = HardCodedClassifier()
model = classifier.fit(X_train, y_train)
y_predicted = model.predict(X_test)

# print("Y_predicted: ", y_predicted)
	# now compare results.
sumCorrect = 0
sumError = 0
for i in range(0,len(y_predicted)):
	if y_predicted[i] == y_test[i]:
		sumCorrect += 1
	else:
		sumError += 1

percentError = 100 * (sumError / (sumError + sumCorrect))

print("Percent Error: {} %".format(percentError))
print("Sum Error: {}".format(sumError))
print("Sum Correct: {}".format(sumCorrect))


# *********** ABOVE AND BEYOND: N-Fold  ***********
# Code adapted from SKLearn docs example
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
# http://scikit-learn.org/stable/modules/cross_validation.html
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True)
print("Number of Splits: {}".format(kf.get_n_splits(iris.data)))

X = iris.data
y = iris.target
for train_index, test_index in kf.split(X_train):
	print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
		
	# *** USE AN EXISTING ALGORITHM TO CREATE A MODEL ***
	from sklearn.naive_bayes import GaussianNB

	classifier = GaussianNB()
	model = classifier.fit(X_train, y_train)


	# *** USE THAT MODEL TO MAKE PREDICTIONS ***
	y_predicted = model.predict(X_test)
	# now compare results.
	sumCorrect = 0
	sumError = 0
	for i in range(0,len(y_predicted)):
		if y_predicted[i] == y_test[i]:
			sumCorrect += 1
		else:
			sumError += 1

	percentError = 100 * (sumError / (sumError + sumCorrect))

	print("Percent Error: {} %".format(percentError))
	print("Sum Error: {}".format(sumError))
	print("Sum Correct: {}".format(sumCorrect))

# *********** ABOVE AND BEYOND: N-Fold, Corrected ***********
# I tried the above code and I got an extra high accuracy on each fold, 
# so I suspect I didn't do it quite right. I sought out another 
# article/documentation/example and am now using that one.
# It makes it even a little easier than before. 
# http://scikit-learn.org/stable/modules/cross_validation.html
from sklearn.model_selection import cross_val_score
from sklearn import svm

clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
print("Scores: {}".format(scores) )



