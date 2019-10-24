# Visualize decision tree
# many types of classifiers
# neural networks
# support vector machine
# lions
# tigers
# bears
# ...

# Why to use decision tree?
# Easy to read and understand. In fact one of few models that are interpretable you can see how the classifier makes a decision

# Goal
# 1. Import dataset

from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
iris
## features
iris.feature_names

## lables
iris.target_names

##
iris.data[0] # returns the measurement for the flower

## contains the lable 0 = setosa
# we're printing out the first row
iris.target[0]


for i in range(len(iris.target)):
    print(f"{iris.target[i]} -> {iris.data[i]}")

## Testing Data
# Examples used to "test" the classifier's accuracy
# Not par of the training data

import numpy as np

# remove one example of each time of flower
test_idx = [0,50,100]

## training data removed 3 entries from the total dataset
# removed some data from the total dataset
train_target = np.delete(iris.target, test_idx)
train_target.shape # (147,)

train_data = np.delete(iris.data, test_idx, axis=0)
train_data.shape #(147, 4)

## testing data
# took the 3 removed entries from a dataset
test_target = iris.target[test_idx]
##
test_data = iris.data[test_idx]

# 2. Train a classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)


# 3. Predict label for new flower
print(test_target) # [0,1,2]
print(clf.predict(test_data)) # splits out the same labels [0,1,2]

# 4. Visualize the tree
##
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf, 
        out_file=dot_data, 
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True, rounded=True,
        impurity=False)

print("should export the tree.dot")

import graphviz as gp
graph = gp.Source(dot_data.getvalue()) 
graph.render("iris", view=True)
