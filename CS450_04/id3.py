

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
# print(iriexis.target_names)

# *********** PREPARE DATA ***********
# used code from example in docs : 
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=52)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

# *********** USE AN EXISTING ALGORITHM TO CREATE A MODEL ***********
# from sklearn.neighbors import KNeighborsClassifier

# classifier = KNeighborsClassifier(n_neighbors=3)
# model = classifier.fit(X_train, y_train)
# predictions = model.predict(X_test)

# from sklearn.metrics import accuracy_score
# acc = accuracy_score(y_test, predictions)
# print("ACCURACY FOR OFF-SHELF KNN : {}".format(acc))




# *********** IMPLEMENT YOUR OWN NEW ALGORITHM ***********

class DecisionNode:
	def __init__(self, featureKey="", children=[], prediction=None, discreteValue=None): # @STRING
		self.featureKey = featureKey # @STRING. The thing in the dataframe that we index into
		self.children = children # @LIST-of-DecisionNodes. of nodes-with-values that represent another branch-off
		self.prediction = prediction
		self.discreteValue = discreteValue

class DecisionTreeClassifier:
	def fit(self, X_train, y_train):
		if len(X_train) != len(y_train):
			return None # Will return error in normal usage. //TODO improve error-handling

		# #DEC Create a dataframe out of the numpy array we receive in
			# BTW We aren't putting the X and y together. We will keep them separate.
		X_df = pd.DataFrame(X_train)
		# y_df = pd.DataFrame(y_train)

		# Create the decision tree
		root = DecisionNode()  # All defaults^
		root = self._makeTree(root, X_df, y_train)

		model = DecisionTreeModel(root)
		return model

	def _makeTree(self, parentNode, X_df, y_train): # Implementation @Recursive
		# @DecisionTreeNode, @DataFrame, @NpArray

		# You SHOULD pass in a default-constructor Node if you're root.
		# If you pass in a None for the node, then you'll get None back... That should be useful later, but IDK.
		if parentNode is None: 
			return None

		# Check to see if....
		# Are we a leaf node? (Have we already determined the prediction we need?)
		if parentNode.prediction is not None:
			return parentNode
			
		# Does y_train have any values in it? # ??? Why would this happen? 
		# I don't think this would happen because if there were only 1 value in the remaining set, 
			#... then we would say that node's prediction is the other value and call it a day, 
			#... make it a leaf node. We wouldn't get to 'this' place


		# Are all y_train rows the same value? (if so, set the node's prediction to that value)
		allYareSame = True
		firstYval = y_train[0]
		for val in y_train:
			if val != firstYval:
				allYareSame = False
		if allYareSame:
			parentNode.prediction = firstYval
			return parentNode

		# Are there no more X_df features/label/columns to choose from? (then we are now a leaf node...)
		# IF there are no more columns/features to choose from.
		if len(X_df.keys()) == 0:
			# THEN Just pick the most common prediction-worthy value.
			yFreqDict = {}
			for val in y_train:
				if val in yFreqDict:
					yFreqDict[val] += 1
				else:
					yFreqDict[val] = 1
			highestFreq = -1
			mostCommonY = y_train[0]
			for key in yFreqDict:
				if yFreqDict[key] > highestFreq:
					highestFreq = yFreqDict[key]
					mostCommonY = key
			parentNode.prediction = mostCommonY
			return parentNode

		######################### SPLITTING TREE #########################
		# For each possible splits... (by each feature / key / column)
		for feature in X_df.keys():
			possibleNode = DecisionNode()
			possibleNode.featureKey = feature

			# Find all the possible discrete values in that feature
			possibleDiscreteValues = {}
			for index, row in X_df.iterrows():
				if row[feature] not in possibleDiscreteValues:
					possibleDiscreteValues[row[feature]] = 0 #arbitrary number so the key is in the dictionary
			
			# Find all the possible target values
			possibleDiscreteTargetValues = {}
			for val in y_train:
				if val not in possibleDiscreteTargetValues:
					possibleDiscreteTargetValues[val] = 0 
			
			# For each possible value ( in that feature / column )
			for possibleDiscreteValue in possibleDiscreteValues: 
				
				# ?? Which indices/rows have that values ....

				# ?? What target label do those indices/rows have?

				# Tally the sums of target values for this value 
				sums = []


				# Get the Entropy value of that possible value

				# (Collect those Entropy values)

			# Take the weighted average of those to get the Entropy of that possible split

			# ? What are the ratios of each branch to the total amount 
			# ? How many 

			# (Collect the Entropy values (per feature) 
			#  AND AND AND the possible node value)

		# Find the highest information gain, or the minimum value in the Entropy Value 
			#... TODO Manage possible difficulty with matching features to entropy values...

		# MAKE THE SPLIT on that feature...
		# make children with their respective discreteValues
		# give the parent have the proper featureKey
		# give the parent references to the children
		# call _makeTreeBelow on the children (as parents) ??? safe?
		# ... give the parent referenes to the children that are 
		# ... returned as valid children nodes (from the _makeTreeBelow function)
		# ... give the _makeTreeBelow function a X_df that has dropped the feature we decided to drop.
		# .... AND AND AND the X_df that only has the indices that have that value we want. .... (drop the rows that aren't in our whitelist)


		# node.featureKey = thatKey
		# node.


		return None
		
	def _entropy(listOfTargetProportions):
		""" Give a list of target proportions for a branch of a feature, 
			or for a possible value of the feature split being investigated
		"""
		# def entrop(a, b): 
		sum = 0
		for targetProportion in listOfTargetProportions:
			a = targetProportion
			sum += (-a * ma.log2(a))
		return sum


class DecisionTreeModel:
	def __init__(self, rootNode):
		self.rootNode = rootNode # @DecisionNode

	def predict(self, X_test):
		n = len(X_test)

		# Transform to a DataFrame
		X_df = pd.DataFrame(X_test)

		predictions = []
		# for each row in the dataframe
		# https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
		for index, row in X_df.iterrows():
			predictions.append(self._predictOne(row, self.rootNode))

		predictions = np.array(predictions)
		return predictions

	def _predictOne(self, row, node):
		# Traverse the tree to predict
		if node.prediction is not None:
			return node.prediction  # target class prediction
		else:
			#.. keep going down the tree
			for child in node.children:
				if row[node.featureKey] == child.discreteValue:
					return _predictOne(row, child) # we only go down the tree where our row's value matches the value of the child's discreteValue

			# ? is children None too?
			if node.children == []:
				return None  # This would be erroneous in the tree-build for prediction and children to be empty


		# If we somehow didn't return before this, this is erroneous, so return None
		return None
	















# *********** RUN OWN NEW ALGORITHM ***********
classifier = DecisionTreeClassifier()
model = classifier.fit(X_train, y_train)
# y_predicted = model.predict(X_test)

# acc = accuracy_score(y_test, y_predicted)
# print("ACCURACY FOR HOME-SPUN KNN : {}".format(acc))












def entropy(listOfTargetProportions):
	""" Give a list of target proportions for a branch of a feature, 
		or for a possible value of the feature split being investigated
	"""
	# def entrop(a, b): 
	sum = 0
	for targetProportion in listOfTargetProportions:
		a = targetProportion
		sum += (-a * ma.log2(a))
	return sum



###### EXPERIMENT #####
dataset = pd.DataFrame([['good', 'high', 'good', 'yes'],
		   ['good', 'high', 'poor', 'yes'],
		   ['good', 'low', 'good', 'yes'],
		   ['good', 'low', 'poor', 'no'],
		   ['average', 'high', 'good', 'yes'],
		   ['average', 'low', 'poor', 'no'],
		   ['average', 'high', 'poor', 'yes'],
		   ['average', 'low', 'good', 'no'],
		   ['low', 'high', 'good', 'yes'],
		   ['low', 'high', 'poor', 'no'],
		   ['low', 'low', 'good', 'no'],
		   ['low', 'low', 'poor', 'no']])

features = dataset[[0,1,2]]
targets = dataset[[3]]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.33, random_state=52)



##### BUILD THE TREE #####
# using X_train

# compare different splits...
# iterate through the features

def buildTree(X_train_df, y_df):
	entropies = []

	# get every possible target value 
	possibleDiscreteTargetValues = []
	targets = y_df.values.tolist()
	for val in targets:
		possibleDiscreteTargetValues.append(val[0])
	possibleDiscreteTargetValues = set(possibleDiscreteTargetValues)

	# Get ready to collect entropies for each feature split
	featureEntropies = [] 

	# for every column/feature there is a possible split
	for column in X_train_df.keys():

		# Get ready to collect entropies for each feature-value branch
		featureValueEntropies = []
		entropyWeights = []
		totalRows = len(X_train_df) # same as len(X_train_df[[column]])

		# every possible value for this column/feature
		possibleValuesForThisColumn = set(X_train_df[column])

		# For each possible value ( in that feature / column )
		for thisPossibleValue in possibleValuesForThisColumn:

			# What indices have that value?
			indices = []
			for idx, row in dataset[[thisPossibleValue]].iterrows():
				if row.values == thisPossibleValue:
					indices.append(idx)
			
			# What target label do those indices/rows have? 
			labels = []
			for idx in indices:
				for arrVal in y_df.values: # goes only once
					labels.append(arrVal[0][idx])

			# Tally the sums of target values for this value
			totalInstances = len(indices)
			sumLabelCounts = {}
			for label in labels:
				if label in sumLabelCounts:
					sumLabelCounts[label] += 1
				else:
					sumLabelCounts[label] = 1
			
			# Get the Entropy value of that possiblevalue
			# ... for every possible target value
			# ... . index into the sumcountlabels dictionary
			# ... . build the list of terms
			# ... . send that to the entropy function
			# ... Save that entropy
			terms = []
			for target in possibleDiscreteTargetValues:
				term = sumLabelCounts[target] / totalInstances
				terms.append(term)
			thisEntropy = entropy(terms)
			featureValueEntropies.append(thisEntropy)

			# Collect Weights
			entropyWeights.append(totalInstances / totalRows) # instances of this possibleValue / 
		
		# Take the weighted average of those to get the Entropy of that possible split
		weightedTerms = []
		sumEntropiesForThisFeatureSplit = 0
		for idx, val in enumerate(entropyWeights):
			thisWeight = entropyWeights[idx]
			thisEntropy = featureValueEntropies[idx]
			sumEntropiesForThisFeatureSplit += thisEntropy * thisWeight
		
		featureEntropies.append( sumEntropiesForThisFeatureSplit )
		# featureEntropies are indexed in sync with the keys of the df

	# Find the highest information gain, lowest entropy value. 
	indexOfBestFeatureSplit = min(featureEntropies)

	#### MAKE THE SPLIT  @@@@ IMPORT 
				
			
					




	# tempList = []
	# for col in dataset: # or dataset.keys()
	# 	for idx, row in dataset[[col]].iterrows():
	# 		tempList.append(idx)
	# print(tempList)



# calculate the entropy of this feature/column and then append it to the 





