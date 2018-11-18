
# Include the LIBSVM package
library(e1071)

# Load our old friend, the Iris data set
# Note that it is included in the default datasets library
library(datasets)
data(iris)

# Partition the data into training and test sets
# by getting a random 30% of the rows as the testRows
allRows <- 1:nrow(iris)
testRows <- sample(allRows, trunc(length(allRows) * 0.3))

# The test set contains all the test rows
irisTest <- iris[testRows,]

# The training set contains all the other rows
irisTrain <- iris[-testRows,]

# Make a cross val set, and reduce the training set a little
oldTrainRows <- 1:nrow(irisTrain)
cvRows <- sample(oldTrainRows, trunc(length(oldTrainRows) * 0.3))
irisCV <- irisTrain[cvRows,]
irisTrain <- irisTrain[-cvRows,]

# Train an SVM model
# Tell it the attribute to predict vs the attributes to use in the prediction,
#  the training data to use, and the kernal to use, along with its hyperparameters.
#  Please note that "Species~." contains a tilde character, rather than a minus

best.accuracy <- 0.0
params.to.try <- c(.0001, .001, .01, .1, 1.0, 10.0, 100.0, 1000.0)
best.g <- params.to.try[1]
best.c <- params.to.try[1]

for (g in params.to.try) {
  for (c in params.to.try) {
    print(paste("Using g:", g))
    print(paste("Using c:", c))
    model <- svm(Species~., data = irisTrain, kernel = "radial", gamma = g, cost = c)
    
    # Use the model to make a prediction on the test set
    # Notice, we are not including the last column here (our target)
    prediction <- predict(model, irisCV[,-5])
    
    # Produce a confusion matrix
    confusionMatrix <- table(pred = prediction, true = irisCV$Species)
    
    # Calculate the accuracy, by checking the cases that the targets agreed
    agreement <- prediction == irisCV$Species
    accuracy <- prop.table(table(agreement))
    
    # Print our results to the screen
    # print(confusionMatrix)
    #print(accuracy)
    
    if (accuracy[2] >= best.accuracy) {
      best.g <- g
      best.c <- c
      best.accuracy <- accuracy[2]
    }
  }
}

# fill in the original full training set
irisTrain <- iris[-testRows,]

# train the model on the full training set
model <- svm(Species~., data = irisTrain, kernel = "radial", gamma = best.g, cost = best.c)

# Use the model to make a prediction on the test set
# Notice, we are not including the last column here (our target)
prediction <- predict(model, irisTest[,-5])

# Produce a confusion matrix
confusionMatrix <- table(pred = prediction, true = irisTest$Species)

# Calculate the test set accuracy, by checking the cases that the targets agreed
agreement <- prediction == irisTest$Species
accuracy <- prop.table(table(agreement))

# Print our full test results to the screen
print(confusionMatrix)
print(accuracy)
print(paste("Best G:", best.g))
print(paste("Best C:", best.c))