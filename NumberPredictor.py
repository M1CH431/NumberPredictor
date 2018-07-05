# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 15:52:23 2018

print "Hello World" 

@author: Michael Ehnes
"""

# Importing libs
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm


# Loading in the digits data set from sklearn
digits = datasets.load_digits()

clf = svm.SVC(gamma = 0.001, C = 100)

# get all of the data until the last element which is the answer
x,y = digits.data[:-25], digits.target[:-25]
clf.fit(x, y)

print('Prediction: ', clf.predict(digits.data)[-18])

plt.imshow(digits.images[-18], cmap = plt.cm.gray_r, interpolation = "nearest")
plt.show()




