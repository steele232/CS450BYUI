
# Include the LIBSVM package
library(e1071)

# Read in our custom dataset
vowel <- read.csv("~/py/CS450BYUI/CS450_09/vowel.csv")

# Partition the data into training and test sets
# by getting a random 20% of the rows as the testRows
allRows <- 1:nrow(vowel)
testRows <- sample(allRows, trunc(length(allRows) * 0.2))

# The test set contains all the test rows
vowelTest <- vowel[testRows,]

# The training set contains all the other rows
vowelTrain <- vowel[-testRows,]

# Make a cross val set, and reduce the training set a little
oldTrainRows <- 1:nrow(vowelTrain)
cvRows <- sample(oldTrainRows, trunc(length(oldTrainRows) * 0.3))
vowelCV <- vowelTrain[cvRows,]
vowelTrain <- vowelTrain[-cvRows,]

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
    model <- svm(Class~., data = vowelTrain, kernel = "radial", gamma = g, cost = c)
    
    # Use the model to make a prediction on the test set
    # Notice, we are not including the last column here (our target)
    prediction <- predict(model, vowelCV[,-13])
    
    # Produce a confusion matrix
    confusionMatrix <- table(pred = prediction, true = vowelCV$Class)
    
    # Calculate the accuracy, by checking the cases that the targets agreed
    agreement <- prediction == vowelCV$Class
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
vowelTrain <- vowel[-testRows,]

# train the model on the full training set
model <- svm(Class~., data = vowelTrain, kernel = "radial", gamma = best.g, cost = best.c)

# Use the model to make a prediction on the test set
# Notice, we are not including the last column here (our target)
prediction <- predict(model, vowelTest[,-13])

# Produce a confusion matrix
confusionMatrix <- table(pred = prediction, true = vowelTest$Class)

# Calculate the test set accuracy, by checking the cases that the targets agreed
agreement <- prediction == vowelTest$Class
accuracy <- prop.table(table(agreement))

# Print our full test results to the screen
print(confusionMatrix)
print(accuracy)
print(paste("Best G:", best.g))
print(paste("Best C:", best.c))