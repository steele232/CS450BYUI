import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from random import shuffle
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn import svm

# 1 Load images into List
imgFileNames = []
rootPicsDir = './Processed/birme28x28/'
for dirName, subdirList, fileList in os.walk(rootPicsDir):
    # print('Found directory: %s' % dirName)
    for fname in fileList:
        imgFileNames.append(fname)
        # img = cv2.imread(rootPicsDir + fname)
        # imgs.append(img)
        print('\t%s' % fname)
 # sort the list of names and then print it out to make sure
 # it's right and then we can move on to index them RIGHT. 
# print("========SORTED========")
imgFileNames.sort()
imgs = []
for name in imgFileNames:
    # print(name)
    if name == ".DS_Store":
        print("FOUND A DS_STORE")
        continue
    img = cv2.imread(rootPicsDir + name, cv2.IMREAD_GRAYSCALE)
    imgs.append(img)

# for img in imgs:
#     cv2.imshow('image',img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



# Flatten each image into a 1 x n matrix, 
# so it can be put into the classifier as 
# each being a feature.
flat_imgs = []
for img in imgs:
    thisImg = []
    for dim1 in img:
        for val in dim1:
            # for val in dim2:
            thisImg.append(val)
    flat_imgs.append(thisImg)
                
# print("len of flat_imgs ", len(flat_imgs))

# flat_np = np.array(flat_imgs)
# print("shape of np ", flat_np.shape)



# Associate them with their 'correct answer'
 # Put them in pairs with their number.
 # 0-16 => 1
 # 17-32 => 2
 # 33-50 => 3
dataPairs = []
x = 0
for image in flat_imgs:
    thisLabel = 1
    if x < 17:
        thisLabel = 1
    if x < 33 and x > 16:
        thisLabel = 2
    if x < 50 and x > 32:
        thisLabel = 3
    dataPairs.append([image, thisLabel, x])
    # print(image[0])
    x = x + 1



# Randomize their order
 # Randomize the list of pairs
shuffle(dataPairs)

# Create a Training Set, CV Set, and Test Set. (30, 10, 12) kill 1
 # Separate and put in the fitting.
features = []
labels = []
num1 = 0
num2 = 0
num3 = 0
x = 0
for pair in dataPairs:
    if x < 30:
        print(pair[1])
    x += 1
    features.append(pair[0])
    labels.append(pair[1])
    if x < 30:
        if pair[1] == 1:
            num1 += 1
        if pair[1] == 2:
            num2 += 1
        if pair[1] == 3:
            num3 += 1

 # grab sublists
trainfeatures = features[0:30]
trainlabels =     labels[0:30]
cvfeatures =    features[30:40]
cvlabels =        labels[30:40]
testfeatures =  features[40:]
testlabels =      labels[40:]

cvPairs = dataPairs[30:40]

# print("train")
# for label in trainlabels:
#     print(label)
# print("cv")
# for label in cvlabels:
#     print(label)
# print("test")
# for label in testlabels:
#     print(label)

# Use a SK Learn model to fit a model, cross validate and test.

clf = tree.DecisionTreeClassifier()
clf = clf.fit(trainfeatures, trainlabels)

# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                     hidden_layer_sizes=(5, 2), random_state=1)
# clf.fit(trainfeatures, trainlabels)

# clf = svm.SVC()
# clf = clf.fit(trainfeatures, trainlabels)


# Cross Validation
 # 
sumCorrect = 0
for i in range(0,10):
    prediction = clf.predict([cvfeatures[i]])
    print("Prediction: ")
    print(prediction)
    actual = cvlabels[i]
    print("Actual: ")
    print(actual)
    print("Index of Actual: ")
    print(cvPairs[i][2])
    if actual == prediction:
        sumCorrect += 1

print (str(sumCorrect) + "/10")
print ("num1 = " + str(num1))
print ("num2 = " + str(num2))
print ("num3 = " + str(num3))


# Testing