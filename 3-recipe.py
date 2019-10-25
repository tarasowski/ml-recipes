# What makes a good feature
# Classifiers are as good as features you provide

import numpy as np
import matplotlib.pyplot as plt

# sample
greyhounds = 500
labs = 500 # labradores

# height is normally distributed
grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

##
help(plt.hist)
##
plt.hist([grey_height, lab_height], stacked=True)
plt.xlabel("height", fontsize=16)
plt.ylabel("# of dogs", fontsize=16)

plt.show()

## height is a useful feature but not precise
# therefore in ML we need multiple features
# how many features and what for a features you need is a more an art than a science
# think about the case if you would be a classifier, which features would you need to classify a dog?

# If a distribution of a feature is about 50 / 50, like down below
# the feature tells us nothing, because it does not correlate with the type of the dog
# avoid useless features they can break your classifier
# your features need to independent (height in inches == height in centimeters) remove it otherwise the machine will double count
# features nees to be easy and understandable
# how many days it will take to mail a letter between cities e.g. a distance between two destinations 2500 miles -> 3 days
# bad feature to use (complex) NYC: 40 N, 74 W, LA: 34 N, 118 W -> 3 days
##
greyh_eyes =  [50, 51]
lab_eyes = [50, 51] 

plt.hist([greyh_eyes, lab_eyes], stacked=True)
plt.show()

