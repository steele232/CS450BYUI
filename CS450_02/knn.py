

import numpy as np
import pandas as pd

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
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=32)
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
from numpy import linalg as LA

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
		self.n_neighbors = n_neighbors

	def predict(self, X_test):
		predictions = []

		# for each element for X_test
		for new_point in X_test:

			# find indices of closest ones.
			distances = []
			isFirst = True
			for old_point in self.X_train:
				# compute the distance between the new_point and old_point
				if isFirst:
					max = LA.norm(old_point - new_point)
					isFirst = False

				distance = LA.norm(old_point - new_point)

				if distance >= max:
					max = distance

				distances.append(distance)
			
			close_classes = []
			for i in range(0,self.n_neighbors):
				min = distances[0]
				minIndex = 0
				for index, item in enumerate(distances):
					if item <= min:
						min = item
						minIndex = index
						distances[index] = max

				
				close_classes.append(y_train[minIndex])
			
			# find the class that is most common
			this_pred = np.bincount(close_classes).argmax()

			# and make it our prediction for this new_point
			predictions.append(this_pred)

		return predictions


# *********** RUN OWN NEW "ALGORITHM" ***********
classifier = KNNClassifier(n_neighbors=3)
model = classifier.fit(X_train, y_train)
y_predicted = model.predict(X_test)

acc = accuracy_score(y_test, y_predicted)
print("ACCURACY FOR HOME-SPUN KNN : {}".format(acc))

# *********** ABOVE AND BEYOND: KD-Tree ***********


