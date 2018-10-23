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





################################################################
#### Autism Spectrum Disorder @ Drop Rows with Missing Values
################################################################

# *********** LOAD DATA ***********
def load_autism_data(should_drop_relation=True):

    should_drop_relation=True
    df = pd.read_csv("./autism-data.csv", 
                    header=None,
                    na_values='?')
    df.dropna(axis=0, inplace=True, how='any') # dropping rows

    y = df[[20]]
    X = df.drop(columns=[20,18])

    if should_drop_relation is True:
        X = X.drop(columns=[19])

    X_np = X.values
    y_np = y.values

    # Encode string target values
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(y_np.ravel())
    label_encoded_y = label_encoder.transform(y_np.ravel())
    label_encoded_y_col = label_encoded_y.reshape(len(label_encoded_y),1)

    # Preprocess the features

    # ###### Features #################
    # 0    # @attribute A1_Score {0,1}
    # 1    # @attribute A2_Score {0,1}
    # 2    # @attribute A3_Score {0,1}
    # 3    # @attribute A4_Score {0,1}
    # 4    # @attribute A5_Score {0,1}
    # 5    # @attribute A6_Score {0,1}
    # 6    # @attribute A7_Score {0,1}
    # 7    # @attribute A8_Score {0,1}
    # 8    # @attribute A9_Score {0,1}
    # 9    # @attribute A10_Score {0,1}
    # 10   # @attribute age numeric
    # 11   # @attribute gender {f,m}
    # 12   # @attribute ethnicity {White-European,Latino,Others,Black,Asian,'Middle Eastern ',Pasifika,'South Asian',Hispanic,Turkish,others}
    # 13   # @attribute jundice {no,yes}
    # 14   # @attribute austim {no,yes}
    # 15   # @attribute contry_of_res {'United States',Brazil,Spain,Egypt,'New Zealand',Bahamas,Burundi,Austria,Argentina,Jordan,Ireland,'United Arab Emirates',Afghanistan,Lebanon,'United Kingdom','South Africa',Italy,Pakistan,Bangladesh,Chile,France,China,Australia,Canada,'Saudi Arabia',Netherlands,Romania,Sweden,Tonga,Oman,India,Philippines,'Sri Lanka','Sierra Leone',Ethiopia,'Viet Nam',Iran,'Costa Rica',Germany,Mexico,Russia,Armenia,Iceland,Nicaragua,'Hong Kong',Japan,Ukraine,Kazakhstan,AmericanSamoa,Uruguay,Serbia,Portugal,Malaysia,Ecuador,Niger,Belgium,Bolivia,Aruba,Finland,Turkey,Nepal,Indonesia,Angola,Azerbaijan,Iraq,'Czech Republic',Cyprus}
    # 16   # @attribute used_app_before {no,yes}
    # 17   # @attribute result numeric
        # 18   # XXX @attribute age_desc {'18 and more'}
    # 19   # @attribute relation {Self,Parent,'Health care professional',Relative,Others} 

    # age-desc is all the same, so it can be dropped
    # relation has many missing values... so it could be dropped, BUT
    # this column seems like it could be very informative, so I don't know if we want to lose that.
    # For the Above and Beyond, I think we will compare one-hot-encoding this value and dropping it altogether


    # ###### TARGET ######################
    # 20   # @attribute Class/ASD {NO,YES}

    # 0-9 binary => fine
    # 10 numeric => fine
    # 11 'm'/'f' -> binary #1
    # 12 label encode / one hot encode [ethnicity]
    # 13 'no'/'yes' -> binary #2
    # 14 'no'/'yes' -> binary #2
    # 15 label encode / one hot encode [country_of_res]
    # 16 'no'/'yes' -> binary #2
    # 17 numeric => fine

    # 19 label encode / one hot encode [relation]
    # 20 Class

    mfLabelEncoder = LabelEncoder()
    mfLabelEncoder = mfLabelEncoder.fit(['m','f'])

    ynLabelEncoder = LabelEncoder()
    ynLabelEncoder = ynLabelEncoder.fit(['yes','no'])


    # Example code:
    # encoded_x = np.concatenate((encoded_x, feature), axis=1)

    # start with the columns that don't need processing
    encoded_x = X_np[:,:11] # columns 0-10

    # Label encode column 11 #1
    label_encoded_x_col = mfLabelEncoder.transform(X_np[:,11]) # get the one column of values 
    label_encoded_x_col = label_encoded_x_col.reshape(X_np.shape[0], 1) # many rows, 1 column
    feature = label_encoded_x_col
    encoded_x = np.concatenate((encoded_x, feature), axis=1)

    # Label encode column 12
    thisLabelEncoder = LabelEncoder()
    thisLabelEncoder = thisLabelEncoder.fit(X_np[:,12])
    label_encoded_x_col = thisLabelEncoder.transform(X_np[:,12]) # get the one column of values 
    label_encoded_x_col = label_encoded_x_col.reshape(X_np.shape[0], 1) # many rows, 1 column
    feature = label_encoded_x_col
    encoded_x = np.concatenate((encoded_x, feature), axis=1)

    # Label encode column 13 w/ #2
    label_encoded_x_col = ynLabelEncoder.transform(X_np[:,13]) # get the one column of values 
    label_encoded_x_col = label_encoded_x_col.reshape(X_np.shape[0], 1) # many rows, 1 column
    feature = label_encoded_x_col
    encoded_x = np.concatenate((encoded_x, feature), axis=1)

    # Label encode column 14 w/ #2
    label_encoded_x_col = ynLabelEncoder.transform(X_np[:,14]) # get the one column of values 
    label_encoded_x_col = label_encoded_x_col.reshape(X_np.shape[0], 1) # many rows, 1 column
    feature = label_encoded_x_col
    encoded_x = np.concatenate((encoded_x, feature), axis=1)

    # Label encode column 15
    thisLabelEncoder = LabelEncoder()
    thisLabelEncoder = thisLabelEncoder.fit(X_np[:,15])
    label_encoded_x_col = thisLabelEncoder.transform(X_np[:,15]) # get the one column of values 
    label_encoded_x_col = label_encoded_x_col.reshape(X_np.shape[0], 1) # many rows, 1 column
    feature = label_encoded_x_col
    encoded_x = np.concatenate((encoded_x, feature), axis=1)

    # Label encode column 16 w/ #2
    label_encoded_x_col = ynLabelEncoder.transform(X_np[:,16]) # get the one column of values 
    label_encoded_x_col = label_encoded_x_col.reshape(X_np.shape[0], 1) # many rows, 1 column
    feature = label_encoded_x_col
    encoded_x = np.concatenate((encoded_x, feature), axis=1)

    # column 17 can be added without processing
    encoded_x = np.concatenate((encoded_x, X_np[:,17].reshape(X_np.shape[0], 1)), axis=1)

    # column 18 is dropped

    # column 19 may be dropped, but maybe label encoded/ one hot encoded
    if should_drop_relation is True:
        pass
    else:
        # label encode it!
        thisLabelEncoder = LabelEncoder()
        thisLabelEncoder = thisLabelEncoder.fit(X_np[:,19])
        label_encoded_x_col = thisLabelEncoder.transform(X_np[:,19]) # get the one column of values 
        label_encoded_x_col = label_encoded_x_col.reshape(X_np.shape[0], 1) # many rows, 1 column
        feature = label_encoded_x_col
        encoded_x = np.concatenate((encoded_x, feature), axis=1)

    # And that's the end!
    return encoded_x, label_encoded_y_col



X, y = load_autism_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=52)

# *********** Evaluations @ SKLearn KNN Classifier ***********

best_cross_val_score = 0
best_k = 1
for k in range(1, 12):
    classifier = KNeighborsClassifier(n_neighbors=k)
    predictions = cross_val_predict(classifier, X_train, y_train.ravel(), cv=10, n_jobs=1) # 10 folds

    acc = accuracy_score(y_train, predictions)
    print("CV ACCURACY for (Autism)  n_neighbors={}: {}".format(k, acc))
    if acc >= best_cross_val_score:
        best_cross_val_score = acc
        best_k = k

print(" ")
print("BEST CV ACCURACY for (Autism) REMOVED[relation] n_neighbors={}: {}".format(best_k, best_cross_val_score))

print(" ")













################################################################
#### Autism Spectrum Disorder @ Drop Rows with Missing Values
################################################################

# *********** LOAD DATA ***********
def load_autism_data_2(should_drop_relation=True):

    should_drop_relation=True
    df = pd.read_csv("./autism-data.csv", 
                    header=None,
                    na_values='?')
    df.dropna(axis=0, inplace=True, how='any') # dropping rows

    y = df[[20]]
    X = df.drop(columns=[20,18,10,12])

    if should_drop_relation is True:
        X = X.drop(columns=[19])

    X_np = X.values
    y_np = y.values

    # Encode string target values
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(y_np.ravel())
    label_encoded_y = label_encoder.transform(y_np.ravel())
    label_encoded_y_col = label_encoded_y.reshape(len(label_encoded_y),1)

    # Preprocess the features

    # ###### Features #################
    # 0    # @attribute A1_Score {0,1}
    # 1    # @attribute A2_Score {0,1}
    # 2    # @attribute A3_Score {0,1}
    # 3    # @attribute A4_Score {0,1}
    # 4    # @attribute A5_Score {0,1}
    # 5    # @attribute A6_Score {0,1}
    # 6    # @attribute A7_Score {0,1}
    # 7    # @attribute A8_Score {0,1}
    # 8    # @attribute A9_Score {0,1}
    # 9    # @attribute A10_Score {0,1}
    # 10   # @attribute age numeric
    # 11   # @attribute gender {f,m}
    # 12   # @attribute ethnicity {White-European,Latino,Others,Black,Asian,'Middle Eastern ',Pasifika,'South Asian',Hispanic,Turkish,others}
    # 13   # @attribute jundice {no,yes}
    # 14   # @attribute austim {no,yes}
    # 15   # @attribute contry_of_res {'United States',Brazil,Spain,Egypt,'New Zealand',Bahamas,Burundi,Austria,Argentina,Jordan,Ireland,'United Arab Emirates',Afghanistan,Lebanon,'United Kingdom','South Africa',Italy,Pakistan,Bangladesh,Chile,France,China,Australia,Canada,'Saudi Arabia',Netherlands,Romania,Sweden,Tonga,Oman,India,Philippines,'Sri Lanka','Sierra Leone',Ethiopia,'Viet Nam',Iran,'Costa Rica',Germany,Mexico,Russia,Armenia,Iceland,Nicaragua,'Hong Kong',Japan,Ukraine,Kazakhstan,AmericanSamoa,Uruguay,Serbia,Portugal,Malaysia,Ecuador,Niger,Belgium,Bolivia,Aruba,Finland,Turkey,Nepal,Indonesia,Angola,Azerbaijan,Iraq,'Czech Republic',Cyprus}
    # 16   # @attribute used_app_before {no,yes}
    # 17   # @attribute result numeric
        # 18   # XXX @attribute age_desc {'18 and more'}
    # 19   # @attribute relation {Self,Parent,'Health care professional',Relative,Others} 

    # age-desc is all the same, so it can be dropped
    # relation has many missing values... so it could be dropped, BUT
    # this column seems like it could be very informative, so I don't know if we want to lose that.
    # For the Above and Beyond, I think we will compare one-hot-encoding this value and dropping it altogether


    # ###### TARGET ######################
    # 20   # @attribute Class/ASD {NO,YES}

    # 0-9 binary => fine
    # 10 numeric => fine
    # 11 'm'/'f' -> binary #1
    # 12 label encode / one hot encode [ethnicity]
    # 13 'no'/'yes' -> binary #2
    # 14 'no'/'yes' -> binary #2
    # 15 label encode / one hot encode [country_of_res]
    # 16 'no'/'yes' -> binary #2
    # 17 numeric => fine

    # 19 label encode / one hot encode [relation]
    # 20 Class

    mfLabelEncoder = LabelEncoder()
    mfLabelEncoder = mfLabelEncoder.fit(['m','f'])

    ynLabelEncoder = LabelEncoder()
    ynLabelEncoder = ynLabelEncoder.fit(['yes','no'])


    # Example code:
    # encoded_x = np.concatenate((encoded_x, feature), axis=1)

    # start with the columns that don't need processing
    encoded_x = X_np[:,:10] # columns 0-10

    # Label encode column 11 #1
    label_encoded_x_col = mfLabelEncoder.transform(X_np[:,10]) # get the one column of values 
    label_encoded_x_col = label_encoded_x_col.reshape(X_np.shape[0], 1) # many rows, 1 column
    feature = label_encoded_x_col
    encoded_x = np.concatenate((encoded_x, feature), axis=1)

    # Label encode column 12
    # thisLabelEncoder = LabelEncoder()
    # thisLabelEncoder = thisLabelEncoder.fit(X_np[:,12])
    # label_encoded_x_col = thisLabelEncoder.transform(X_np[:,12]) # get the one column of values 
    # label_encoded_x_col = label_encoded_x_col.reshape(X_np.shape[0], 1) # many rows, 1 column
    # feature = label_encoded_x_col
    # encoded_x = np.concatenate((encoded_x, feature), axis=1)


    # Label encode column 13 w/ #2
    label_encoded_x_col = ynLabelEncoder.transform(X_np[:,11]) # get the one column of values 
    label_encoded_x_col = label_encoded_x_col.reshape(X_np.shape[0], 1) # many rows, 1 column
    feature = label_encoded_x_col
    encoded_x = np.concatenate((encoded_x, feature), axis=1)

    # Label encode column 14 w/ #2
    label_encoded_x_col = ynLabelEncoder.transform(X_np[:,12]) # get the one column of values 
    label_encoded_x_col = label_encoded_x_col.reshape(X_np.shape[0], 1) # many rows, 1 column
    feature = label_encoded_x_col
    encoded_x = np.concatenate((encoded_x, feature), axis=1)

    # Label encode column 15
    thisLabelEncoder = LabelEncoder()
    thisLabelEncoder = thisLabelEncoder.fit(X_np[:,13])
    label_encoded_x_col = thisLabelEncoder.transform(X_np[:,13]) # get the one column of values 
    label_encoded_x_col = label_encoded_x_col.reshape(X_np.shape[0], 1) # many rows, 1 column
    feature = label_encoded_x_col
    encoded_x = np.concatenate((encoded_x, feature), axis=1)

    # Label encode column 16 w/ #2
    label_encoded_x_col = ynLabelEncoder.transform(X_np[:,14]) # get the one column of values 
    label_encoded_x_col = label_encoded_x_col.reshape(X_np.shape[0], 1) # many rows, 1 column
    feature = label_encoded_x_col
    encoded_x = np.concatenate((encoded_x, feature), axis=1)

    # column 17 can be added without processing
    encoded_x = np.concatenate((encoded_x, X_np[:,15].reshape(X_np.shape[0], 1)), axis=1)

    # column 18 is dropped

    # column 19 may be dropped, but maybe label encoded/ one hot encoded
    if should_drop_relation is True:
        pass
    else:
        # label encode it!
        thisLabelEncoder = LabelEncoder()
        thisLabelEncoder = thisLabelEncoder.fit(X_np[:,17])
        label_encoded_x_col = thisLabelEncoder.transform(X_np[:,17]) # get the one column of values 
        label_encoded_x_col = label_encoded_x_col.reshape(X_np.shape[0], 1) # many rows, 1 column
        feature = label_encoded_x_col
        encoded_x = np.concatenate((encoded_x, feature), axis=1)

    # And that's the end!
    return encoded_x, label_encoded_y_col



X, y = load_autism_data_2()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=52)

# *********** Evaluations @ SKLearn KNN Classifier ***********

best_cross_val_score = 0
best_k = 1
for k in range(1, 12):
    classifier = KNeighborsClassifier(n_neighbors=k)
    predictions = cross_val_predict(classifier, X_train, y_train.ravel(), cv=10, n_jobs=1) # 10 folds

    acc = accuracy_score(y_train, predictions)
    print("CV ACCURACY for (Autism) n_neighbors={}: {}".format(k, acc))
    if acc >= best_cross_val_score:
        best_cross_val_score = acc
        best_k = k

print(" ")
print("BEST CV ACCURACY for (Autism) REMOVED[ethnicity, country_of_res, relation] n_neighbors={}: {}".format(best_k, best_cross_val_score))

print(" ")


