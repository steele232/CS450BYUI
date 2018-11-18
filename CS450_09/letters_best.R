
# Include the LIBSVM package
library(e1071)

# Read in our custom dataset
letters <- read.csv("~/py/CS450BYUI/CS450_09/letters.csv")

# Partition the data into training and test sets
# by getting a random 20% of the rows as the testRows
allRows <- 1:nrow(letters)
testRows <- sample(allRows, trunc(length(allRows) * 0.2))

# The test set contains all the test rows
lettersTest <- letters[testRows,]

# The training set contains all the other rows
lettersTrain <- letters[-testRows,]


# Train an SVM model
best.g <- 0.1
best.c <- 10.0

# train the model on the full training set
model <- svm(letter~., data = lettersTrain, kernel = "radial", gamma = best.g, cost = best.c)

# Use the model to make a prediction on the test set
# Notice, we are not including the last column here (our target)
prediction <- predict(model, lettersTest[,-1])

# Produce a confusion matrix
confusionMatrix <- table(pred = prediction, true = lettersTest$letter)

# Calculate the test set accuracy, by checking the cases that the targets agreed
agreement <- prediction == lettersTest$letter
accuracy <- prop.table(table(agreement))

# Print our full test results to the screen
print(confusionMatrix)
print(accuracy)
print(paste("Best G:", best.g))
print(paste("Best C:", best.c))