

import numpy as np



# *********** LOAD DATA ***********
from sklearn import datasets
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
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=42)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

# *********** USE AN EXISTING ALGORITHM TO CREATE A MODEL ***********
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=3)
model = classifier.fit(X_train, y_train)
predictions = model.predict(X_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, predictions)
print("ACCURACY FOR OFF-SHELF KNN : {}".format(acc))




# *********** IMPLEMENT YOUR OWN NEW "ALGORITHM" ***********

class KNNClassifier:
	def __init__(self, n_neighbors=3):
		self.n_neighbors = n_neighbors
	def fit(self, X_train, y_train):
		model = KNNModel(X_train, y_train, self.n_neighbors)
		return model

class KNNModel:
	def __init__(self, X_train, y_train, n_neighbors):
		self.X_train = X_train
		self.y_train = y_train
	def predict(self, data_test):
		# find indices of closest ones.

		# sort X_train by how far they are from point[0]

		# 

		length = len(data_test)
		predictions = np.zeros(length)
		return predictions


# *********** RUN OWN NEW "ALGORITHM" ***********

classifier = KNNClassifier(n_neighbors=3)
model = classifier.fit(X_train, y_train)
y_predicted = model.predict(X_test)

acc = accuracy_score(y_test, y_predicted)
print("ACCURACY FOR ROLL-YER-OWN KNN : {}".format(acc))


# *********** ABOVE AND BEYOND: KD-Tree ***********


