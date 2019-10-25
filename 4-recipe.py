# Write a pipeline for supervised learning


# partiton the datasets into two parts: train and test

from sklearn import datasets
iris = datasets.load_iris()

# f(x) = y
# classifier is a function at a high lever x is the input and y is the output
# f(x) == feature
# y == label
X = iris.data
y = iris.target

# partition the dataset into train and test
from sklearn.model_selection import train_test_split
# half of the data will be in train and half of the data will be in test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .5)

# create classifier
# an example for a classifier from a high level perspective
# def classify(features):
#   `model` defines the rules of our function -> do some logic
#   `model` has parameters that we can adjust with our training data
#   return label

from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
print(predictions)

# calculate the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)
