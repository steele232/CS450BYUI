import numpy as np
import pandas as pd
import math as ma

# *********** LOAD DATA ***********
from sklearn import datasets
iris = datasets.load_iris()


### Let's discretize. ...


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
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=52)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)



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
		y_df = pd.DataFrame(y_train)


		# Create the decision tree
		root = DecisionNode()  # All defaults^
		root = self._makeTree(root, X_df, y_df)

		model = DecisionTreeModel(root)
		return model

	def _makeTree(self, parentNode, X_df, y_df): # Implementation @Recursive
		# @DecisionTreeNode, @DataFrame, @NpArray
		print("*** _makeTree called")

		# You SHOULD pass in a default-constructor Node if you're root.
		# If you pass in a None for the node, then you'll get None back... That should be useful later, but IDK.
		if parentNode is None: 
			return None

		# Check to see if....
		# Are we a leaf node? (Have we already determined the prediction we need?)
		if parentNode.prediction is not None:
			return parentNode

		# Are all y_df rows the same value? (if so, set the node's prediction to that value)
		allYareSame = True
		isFirstTime = True
		# DO THIS THE PANDAS WAY.
		for idx, val in y_df.iterrows():
			if isFirstTime:
				firstYval = val
				isFirstTime = False
			else:
				if firstYval is not val:
					allYareSame = False
		if allYareSame:
			parentNode.prediction = firstYval.values
			print("allYareSame!!")
			print(firstYval.values)
			return parentNode

		# Are there no more X_df features/label/columns to choose from? (then we are now a leaf node...)
		# IF there are no more columns/features to choose from.
		if len(X_df.keys()) == 0:
			# THEN Just pick the most common prediction-worthy value.
			yFreqDict = {}
			for val in y_df:
				if val in yFreqDict:
					yFreqDict[val] += 1
				else:
					yFreqDict[val] = 1
			highestFreq = -1
			mostCommonYval = ""
			isFirstTime = True
			for key in yFreqDict:
				if isFirstTime:
					mostCommonYval = key
					isFirstTime = False
				if yFreqDict[key] > highestFreq:
					highestFreq = yFreqDict[key]
					mostCommonYval = key
			parentNode.prediction = mostCommonYval
			print("out of features! mostCommonYval!")
			print(mostCommonYval)
			return parentNode


		### SETUP #####
		###############

		# get every possible target value 
		possibleDiscreteTargetValues = []
		targets = y_df.values.tolist()
		for val in targets:
			possibleDiscreteTargetValues.append(val[0])
		possibleDiscreteTargetValues = list(set(possibleDiscreteTargetValues))

		# Get ready to collect entropies for each feature split
		featureEntropies = []
		possibleFeatures = []
		featureListOfListOfPossibleValues = []

		######################### SPLITTING TREE #########################
		# For each possible splits... (by each feature / key / column)
		for feature in X_df.keys():

			# Save this featureKey in case this is the best split
			thisFeatureKey = feature

			# Get ready to collect entropies for each feature-value branch
			featureValueEntropies = []
			entropyWeights = []
			possibleValuesInFeature = []
			totalRows = len(X_df) # same as len(X_df[[column]])

			# every possible value for this column/feature
			possibleValuesForThisFeature = list(set(X_df[feature]))
			featureListOfListOfPossibleValues.append( possibleValuesForThisFeature )

			# For each possible value ( in that feature / column )
			for thisPossibleValue in possibleValuesForThisFeature: 


				
				
				# What indices have that value?
				# I will get an array of indices to work with
				indices = []
				for idx, row in X_df[[thisFeatureKey]].iterrows():
					# print("*********** ROW ***********")
					# print(row.values)
					# print(idx)
					if row.values == thisPossibleValue:
						# print(row.values)
						# print(idx)
						indices.append(idx)

				# What target label do those indices/rows have? 
				# I will get an array of target values, which indexes will match the indices array
				labels = []
				for idx in indices:

					print("*********** ROW ***********")
					print(y_df.iloc[idx])
					print(y_df.iloc[idx].values.tolist()[0])
					labels.append(y_df.iloc[idx].values.tolist()[0])

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
				thisEntropy = self._entropy(terms)
				featureValueEntropies.append(thisEntropy)

				# Collect Weights
				entropyWeights.append(totalInstances / totalRows) # instances of this possibleValue / 
			
			###
			# Take the weighted average of those to get the Entropy of that possible split
			weightedTerms = []
			sumEntropiesForThisFeatureSplit = 0
			for idx, val in enumerate(entropyWeights):
				thisWeight = entropyWeights[idx]
				thisEntropy = featureValueEntropies[idx]
				sumEntropiesForThisFeatureSplit += thisEntropy * thisWeight
			# (Collect the Entropy values (per feature)
			featureEntropies.append( sumEntropiesForThisFeatureSplit )
			#  AND AND AND the possible node value)
			possibleFeatures.append( thisFeatureKey )


		###
		# Find the highest information gain, lowest entropy value. 
		indexOfBestFeatureSplit = np.argmin(featureEntropies)
		featureWeAreSplittingOn = possibleFeatures[indexOfBestFeatureSplit]

		# Assign the featureKey of the parentNode
		parentNode.featureKey = featureWeAreSplittingOn

		allValuesForThisFeature = featureListOfListOfPossibleValues[indexOfBestFeatureSplit]

		# Create children nodes for each possible discreteValue in the feature
		children = []
		for value in allValuesForThisFeature:
			# and put their discreteValues in them
			newChild = DecisionNode(discreteValue=value)
			children.append(newChild)

		# Iterate through the children/featureValues (in a pass-by-reference-ready way)
		for i in range(len(children)):

			# NOTE: len( allValuesForThisFeature ) == len( children )

			##### SLICE # Get a sliced up version of the data (get ready to recur)

			# Let's slice the y_df first
			y_df_sliced = y_df.loc[X_df[featureWeAreSplittingOn] == children[i].discreteValue]


			# Then let's slice the X_df
			X_df_sliced = X_df.loc[X_df[featureWeAreSplittingOn] == children[i].discreteValue]
			# Then drop the column
			X_df_sliced = X_df_sliced.drop(columns=[featureWeAreSplittingOn])

			# Call the _createTree recur-function on each child. (child, w/ respective dataset)
			children[i] = self._makeTree(children[i], X_df_sliced, y_df_sliced)

		
		# Attach the Children to the parentNode
		parentNode.children = children

		############
		# CHECKSUM #
		############
		# IF we've made it this far.
		# the ParentNode we return
		# should:
		# ... have a new featureKey
		# ... have new children nodes
		# ... not have a prediction

		return parentNode
		
	def _entropy(self, listOfTargetProportions):
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
			predictions.append([self._predictOne(row, self.rootNode)])

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
					return self._predictOne(row, child) # we only go down the tree where our row's value matches the value of the child's discreteValue

			# ? is children None too?
			if node.children == []:
				return None  # This would be erroneous in the tree-build for prediction and children to be empty


		# If we somehow didn't return before this, this is erroneous too, so return None
		return None
	









# Use In-class dataset.
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

X_train, X_test, y_train, y_test = train_test_split(dataset[[0,1,2]], dataset[[3]], test_size=0.20, random_state=56)


# Use Voting Dataset
dataset = pd.read_csv("./house-votes-84.csv", header=None, true_values='y', false_values='n', na_values='?')
X_train, X_test, y_train, y_test = train_test_split(dataset.drop(columns=[0]), dataset[[0]], test_size=0.20, random_state=56)




# *********** USE AN EXISTING ALGORITHM TO CREATE A MODEL ***********
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=3)
model = classifier.fit(X_train, y_train)
predictions = model.predict(X_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, predictions)
print("ACCURACY FOR OFF-SHELF KNN : {}".format(acc))





# *********** RUN OWN NEW ALGORITHM ***********

classifier = DecisionTreeClassifier()
model = classifier.fit(X_train, y_train)
y_predicted = model.predict(X_test)

print("************************")
print("Made it past predict")
print("************************")
print(y_predicted)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predicted)
print("ACCURACY FOR HOME-SPUN KNN : {}".format(acc))










