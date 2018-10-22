import numpy as np
import pandas as pd
import math as ma


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict


################################################################
#### Car Evaluation
################################################################

# *********** LOAD DATA ***********
def load_Car_Evaluation_Data():
    """ Returns X, y as numpy arrays / arrays-of-arrays.
        Loads up the Car Evaluation Dataset from this directory.
        Handles all the preprocessing logic. 
    """

    # read it in (default)
    df = pd.read_csv("./car.csv", header=None)

    # Separate features and targets
    X = df.drop(columns=[6])
    y = df[[6]]

    X_np = X.values
    y_np = y.values

    # Encode string target values as integers
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(y_np.ravel())
    label_encoded_y = label_encoder.transform(y_np.ravel())
    label_encoded_y_col = label_encoded_y.reshape(len(label_encoded_y),1)

    # #Decision. We will label encode all these 
    # attributes because they all have ordinal qualities

    # KEY :
    # buying       v-high, high, med, low
    # maint        v-high, high, med, low
    # doors        2, 3, 4, 5-more
    # persons      2, 4, more
    # lug_boot     small, med, big
    # safety       low, med, high

    # Make and use a unique label encoder 
    # for every single question/column
    # Then concatenate it to the encoded_x
    encoded_x = None
    for i in range(0, X_np.shape[1]):

        # Make an encoder specific to this column
        label_x_encoder = LabelEncoder()
        label_x_encoder = label_x_encoder.fit(X_np[:,i]) # get the one column of values

        # Label Encode them
        label_encoded_x_col = label_x_encoder.transform(X_np[:,i]) # get the one column of values 
        label_encoded_x_col = label_encoded_x_col.reshape(X_np.shape[0], 1) # many rows, 1 column
        feature = label_encoded_x_col

        # One Hot Encode them
        # onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
        # feature = onehot_encoder.fit_transform(label_encoded_x_col)

        if encoded_x is None:
            encoded_x = feature
        else:
            encoded_x = np.concatenate((encoded_x, feature), axis=1)

    return encoded_x, label_encoded_y_col


X, y = load_Car_Evaluation_Data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=52)

# *********** Evaluations @ SKLearn KNN Classifier ***********

best_cross_val_score = 0
for k in range(1, 12):
    classifier = KNeighborsClassifier(n_neighbors=k)
    predictions = cross_val_predict(classifier, X_train, y_train.ravel(), cv=10, n_jobs=1) # 10 folds

    best_k = k
    acc = accuracy_score(y_train, predictions)
    print("CV ACCURACY for (Car Evaluations) n_neighbors={}: {}".format(k, acc))
    if acc >= best_cross_val_score:
        best_cross_val_score = acc
        best_k = k

print(" ")
print("BEST CV ACCURACY for (Car Evaluations) n_neighbors={}: {}".format(best_k, best_cross_val_score))


print(" ")


################################################################
#### Autism Spectrum Disorder
################################################################

# *********** LOAD DATA ***********
def load_autism_data():
    pass



# X, y = load_autism_data()

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=52)

# # *********** Evaluations @ SKLearn KNN Classifier ***********

# best_cross_val_score = 0
# best_k = 1
# for k in range(1, 12):
#     classifier = KNeighborsClassifier(n_neighbors=k)
#     predictions = cross_val_predict(classifier, X_train, y_train.ravel(), cv=10, n_jobs=1) # 10 folds

#     acc = accuracy_score(y_train, predictions)
#     print("CV ACCURACY for (Autism) n_neighbors={}: {}".format(k, acc))
#     if acc >= best_cross_val_score:
#         best_cross_val_score = acc
#         best_k = k

# print(" ")
# print("BEST CV ACCURACY for (Autism) n_neighbors={}: {}".format(best_k, best_cross_val_score))

# print(" ")















################################################################
#### Auto-Mpg 
################################################################
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

dataset = pd.read_csv("./auto-mpg.csv", header=None, 
                        delim_whitespace=True, na_values="?")
dataset.dropna(axis=0, inplace=True, how='any') # dropping rows
X = dataset.drop(columns=[0,8]) # take out targets, 
                                # and column with unique strings
y = dataset[[0]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=52)

best_cross_val_score = 10000
best_k = 1
for k in range(1,15):
    classifier = KNeighborsRegressor(n_neighbors=k)
    model = classifier.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metric = mean_absolute_error(y_test, predictions)

    print("CV ACCURACY for (Auto-Mpg) n_neighbors={}: {}".format(k, metric))
    if metric <= best_cross_val_score:
        best_cross_val_score = metric
        best_k = k

print(" ")
print("BEST CV ACCURACY for (Auto-Mpg) n_neighbors={}: {}".format(best_k, best_cross_val_score))

print(" ")







