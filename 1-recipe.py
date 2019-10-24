# What is a machine learning?
print("hello machine")

# ML is a study of algorithms that learn from examples and experience instead of learning from hard-coded rules


# We need to find the distintion between apple and orange
# For that we need an algorithm that is called a classifier
# Think of classifier as a function, it takes some data as input and return an output data with a label
# The technique to write a classfier is called supervised learning:
# it begins with examples of the problem you want solve

from sklearn import tree
# supervised learning
# step 1: collect training data -> these are examples of the problem we want to solve, for our problem we write a function that classifies a piece of fruit, it will provide description if its a fruit based on features like a texture. To collect our data we need to make measurement that describe them in a table weight | texture | label, in ML these measurements are called features. To keep it simple we use only two features weigth in grams and texture. Each row in our training data is an EXAMPLE.The last column is called a label, it identifies which type of fruit is in each row, there are currently two possibilies orange, apple. The whole table is our training data, the classifier can learn from. 

# features is the input for the classifier
# use 0 for bumpy and 1 for smooth
features = [[140, "smooth"], [130, "smooth"], [150, "bumpy"], [170, "bumpy"]]

def convert_ft(features):
    return [(lambda xs: [xs[0], 0] if xs[1] == "bumpy" else [xs[0], 1])(ar)
            for ar in features
            ]

features_ = convert_ft(features)
print(features_)

##
# lables as the output for the classifier that we want
# 0 for apple and 1 for orange
labels = ["apple", "apple", "orange", "orange"]

def convert_lb(labels):
    return [(lambda x: 0 if x == "apple" else 1)(ar)
            for ar in labels
            ]

labels_ = convert_lb(labels)
labels_

# step 2: train classifier -> to use the example from step 1 to train the classifier. The type of classifier we will start with is called a decision tree
# create the classifier
# at this point it's just an empty box of rules
# tree.DecisionTreeClassifier() it doesn't know anything about apples and oranges yet
# to train a classifier we need a learning algorithm to find patterns in the data
clf = tree.DecisionTreeClassifier()

# fit is a synonym for find patterns in data
clf = clf.fit(features_, labels_)

# step 3: make predictions -> we'll use the classifier to classify a new fruit
# the input for the classifier is the features for a new example
# 150 grams and bumpy
# the output will 0 if it's an apple or 1 if it's an orange
clf.predict([[150, 0]]) # returns 1

# TODO: Create a new classifier for a new problem just by changing the training data

feature_cars = [[300, 2], [450, 2], [200, 8], [150, 9]]
# 0 = sports-car
# 1 = minivan
label_cars = [0, 0, 1, 1]

clf_cars = clf.fit(feature_cars, label_cars)
clf_cars.predict([[500, 2]]) # 0
clf_cars.predict([[200, 1]]) # 0
clf_cars.predict([[200, 2]]) # 0
clf_cars.predict([[200, 9]]) # 1
