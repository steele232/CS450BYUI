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
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
model = dtree.fit(X_train, y_train)
predictions = model.predict(X_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, predictions)
print(" ")
print("ACCURACY ==> IRIS @ DEFAULT, AS-IS @ SKLearn Decision Tree : {}".format(acc))
print(" ")

# *********** VISUALIZE ***********
from sklearn import tree
from subprocess import call
tree.export_graphviz(model, out_file='cooltree.dot')
call(["dot", "-Tpng", "cooltree.dot", "-o", "./viz/iris_tree_1.png"])







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

    dtree = DecisionTreeClassifier()
    model = dtree.fit(X_train, y_train)
    predictions = model.predict(X_test)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, predictions)
    print("ACCURACY ==> IRIS @ BINNING with " + str(i) + " bins @ SKLearn Decision Tree : {}".format(acc))
print(" ")

# *********** VISUALIZE ***********
from sklearn import tree
from subprocess import call
tree.export_graphviz(dtree, out_file='cooltree.dot')
call(["dot", "-Tpng", "cooltree.dot", "-o", "./viz/iris_tree_2.png"])




################################################################
#### VOTING @ Removing Missing Rows @ SKLearn Decision Tree
################################################################
dataset = pd.read_csv("./house-votes-84.csv", header=None, true_values='y', false_values='n', na_values='?')
dataset.dropna(axis=0, inplace=True, how='any') # dropping rows
X = dataset.drop(columns=[0])
y = dataset[[0]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=52)

classifier = DecisionTreeClassifier()
model = classifier.fit(X_train, y_train)
predictions = model.predict(X_test)

acc = accuracy_score(y_test, predictions)
print("ACCURACY ==> VOTING @ Removing Missing Rows @ SKLearn Decision Tree : {}".format(acc))
print(" ")

tree.export_graphviz(model, out_file='cooltree.dot')
call(["dot", "-Tpng", "cooltree.dot", "-o", "./viz/voting_remove_missing_rows.png"])


################################################################
#### VOTING @ Removing Missing Columns @ SKLearn Decision Tree
################################################################
dataset = pd.read_csv("./house-votes-84.csv", header=None, true_values='y', false_values='n', na_values='?')
dataset.dropna(axis=1, inplace=True, how='any') # dropping columns
X = dataset.drop(columns=[0])
y = dataset[[0]]

print("ACCURACY ==> VOTING @ Removing Missing Columns @ SKLearn Decision Tree : ????")
print("Length of X : {}".format(len(X)))
print("Shape of X : {}".format(str(X.values.shape)))
print("Length of y : {}".format(len(y)))
print("Shape of y : {}".format(str(y.values.shape)))
print("--> !!! If you remove columns with any '?' in them, then you have 0 columns left. ")
print(" ")





################################################################
#### VOTING @ ONE HOT ENCODING @ SKLearn Decision Tree
################################################################
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# *********** @VOTING @SKLearn Decision Tree ***********
dataset = pd.read_csv("./house-votes-84.csv", header=None)
X = dataset.drop(columns=[0])
y = dataset[[0]]

X_np = X.values
y_np = y.values

# Encode string class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y_np.ravel())
label_encoded_y = label_encoder.transform(y_np.ravel())
label_encoded_y_col = label_encoded_y.reshape(len(label_encoded_y),1)

saved_y_label_encoder = label_encoder

# we can use the x label encoder for every single question/column
possible_X_values = np.array(['y', 'n', '?'])
label_x_encoder = LabelEncoder()
label_x_encoder = label_x_encoder.fit(possible_X_values)

encoded_x = None
for i in range(0, X_np.shape[1]):

    # Label Encode them
    label_encoded_x_col = label_x_encoder.transform(X_np[:,i]) # get the one column of values 
    label_encoded_x_col = label_encoded_x_col.reshape(X_np.shape[0], 1) # many rows, 1 column

    # One Hot Encode them
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    feature = onehot_encoder.fit_transform(label_encoded_x_col)

    if encoded_x is None:
        encoded_x = feature
    else:
        encoded_x = np.concatenate((encoded_x, feature), axis=1)

# print("X shape: : ", encoded_x.shape)

X_train, X_test, y_train, y_test = train_test_split(encoded_x, label_encoded_y_col, test_size=0.20, random_state=52)

classifier = DecisionTreeClassifier()
model = classifier.fit(X_train, y_train)
predictions = model.predict(X_test)

acc = accuracy_score(y_test, predictions)
print("ACCURACY ==> VOTING @ ONE HOT ENCODING @ SKLearn Decision Tree : {}".format(acc))
print(" ")

tree.export_graphviz(model, out_file='cooltree.dot')
call(["dot", "-Tpng", "cooltree.dot", "-o", "./viz/voting_one_hot.png"])




################################################################
#### VOTING @ PRUNING (w/1-HOT) @ SKLearn Decision Tree
################################################################
# NOTE: We are going to use cross_val_predict for pruning options
# so that we aren't tuning the algorithm on the test set.
from sklearn.model_selection import cross_val_predict

best_cross_val_score = 0

for thisDepth in range(3, 15):
    classifier = DecisionTreeClassifier(max_depth=thisDepth, random_state=42)
    predictions = cross_val_predict(classifier, X_train, y_train, cv=5, n_jobs=1) # 5 folds
    model = classifier.fit(X_train, y_train) # for tree visualizing

    acc = accuracy_score(y_train, predictions)
    print("CV ACCURACY ==> VOTING @ PRUNING Max Depth: {} (w/1-HOT) @ SKLearn Decision Tree : {}".format(thisDepth, acc))
    if acc >= best_cross_val_score:
        best_cross_val_score = acc

    tree.export_graphviz(model, out_file='cooltree.dot')
    call(["dot", "-Tpng", "cooltree.dot", "-o", "./viz/voting_oh_dtree_pruned_depth_{}.png".format(thisDepth)])

print(" ")

for thisManyLeaves in range(3, 25):
    classifier = DecisionTreeClassifier(max_leaf_nodes=thisManyLeaves, random_state=42)
    predictions = cross_val_predict(classifier, X_train, y_train, cv=5, n_jobs=1) # 5 folds
    model = classifier.fit(X_train, y_train) # for tree visualizing
    if acc >= best_cross_val_score:
        best_cross_val_score = acc

    acc = accuracy_score(y_train, predictions)
    print("CV ACCURACY ==> VOTING @ PRUNING Max Leaves: {} (w/1-HOT) @ SKLearn Decision Tree : {}".format(thisManyLeaves, acc))

    tree.export_graphviz(model, out_file='cooltree.dot')
    call(["dot", "-Tpng", "cooltree.dot", "-o", "./viz/voting_oh_dtree_pruned_leaves_{}.png".format(thisManyLeaves)])



print(" ")




################################################################
#### VOTING @ Pruning w/ Removing Missing Rows @ SKLearn Decision Tree
################################################################
dataset = pd.read_csv("./house-votes-84.csv", header=None, true_values='y', false_values='n', na_values='?')
dataset.dropna(axis=0, inplace=True, how='any') # dropping rows
X = dataset.drop(columns=[0])
y = dataset[[0]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=52)

for thisDepth in range(3, 15):
    classifier = DecisionTreeClassifier(max_depth=thisDepth, random_state=42)
    predictions = cross_val_predict(classifier, X_train, y_train, cv=5, n_jobs=1) # 5 folds
    model = classifier.fit(X_train, y_train) # for tree visualizing
    if acc >= best_cross_val_score:
        best_cross_val_score = acc

    acc = accuracy_score(y_train, predictions)
    print("CV ACCURACY ==> VOTING @ PRUNING Max Depth: {} (w/ Removing Missing Rows) @ SKLearn Decision Tree : {}".format(thisDepth, acc))

    tree.export_graphviz(model, out_file='cooltree.dot')
    call(["dot", "-Tpng", "cooltree.dot", "-o", "./viz/voting_dtree_pruned_depth_remove_rows_{}.png".format(thisDepth)])

print(" ")

for thisManyLeaves in range(3, 25):
    classifier = DecisionTreeClassifier(max_leaf_nodes=thisManyLeaves, random_state=42)
    predictions = cross_val_predict(classifier, X_train, y_train, cv=5, n_jobs=1) # 5 folds
    model = classifier.fit(X_train, y_train) # for tree visualizing
    if acc >= best_cross_val_score:
        best_cross_val_score = acc

    acc = accuracy_score(y_train, predictions)
    print("CV ACCURACY ==> VOTING @ PRUNING Max Leaves: {} (w/ Removing Missing Rows) @ SKLearn Decision Tree : {}".format(thisManyLeaves, acc))

    tree.export_graphviz(model, out_file='cooltree.dot')
    call(["dot", "-Tpng", "cooltree.dot", "-o", "./viz/voting_dtree_pruned_leaves_{}.png".format(thisManyLeaves)])



print(" ")




################################################################
#### VOTING @ FINAL Pruning w/ Removing Missing Rows @ SKLearn Decision Tree
################################################################
print("Best best_cross_val_score: {}".format(best_cross_val_score)) # .978 accuracy 
print("As of this writing: This came from Removing Missing Rows, and using a Max Depth of 4")
print(" ")
print("So we will use this configuration for testing on the test set, the final test. Here it goes!!")

dataset = pd.read_csv("./house-votes-84.csv", header=None, true_values='y', false_values='n', na_values='?')
dataset.dropna(axis=0, inplace=True, how='any') # dropping rows
X = dataset.drop(columns=[0])
y = dataset[[0]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=52)

classifier = DecisionTreeClassifier(max_depth=4, random_state=42)
model = classifier.fit(X_train, y_train)
predictions = model.predict(X_test)

acc = accuracy_score(y_test, predictions)
print("FINAL TEST ACCURACY ==> VOTING @ Removing Missing Rows, Max_Depth=4 @ SKLearn Decision Tree : {}".format(acc))
print(" ")

tree.export_graphviz(model, out_file='cooltree.dot')
call(["dot", "-Tpng", "cooltree.dot", "-o", "./viz/voting_final_test.png"])







################################################################
#### AUTO-MPG @ SKLearn Decision Tree Regressor
################################################################
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


dataset = pd.read_csv("./auto-mpg.csv", header=None, 
                        delim_whitespace=True, na_values="?")
dataset.dropna(axis=0, inplace=True, how='any') # dropping rows
X = dataset.drop(columns=[0,8]) # take out targets, 
                                # and column with unique strings
y = dataset[[0]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=52)

classifier = DecisionTreeRegressor(random_state=42)
model = classifier.fit(X_train, y_train)
predictions = model.predict(X_test)

metric = mean_absolute_error(y_test, predictions)
print(" ")
print("ACCURACY ==> AUTO-MPG @ SKLearn Decision Tree Regressor: {}".format(metric))
print("I believe this means that the regression model we made ")
print("is, on average, within +/- 2 mpg of the actual ")
print(" ")

tree.export_graphviz(model, out_file='cooltree.dot')
call(["dot", "-Tpng", "cooltree.dot", "-o", "./viz/auto.png"])

















print(" ")
call(["rm", "cooltree.dot"])

