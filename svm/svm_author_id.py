#!/usr/bin/python

"""
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors

    Sara has label 0
    Chris has label 1

"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################

clf = SVC(kernel='linear', C=1.0)

t0 = time() #timining time to train
clf.fit(features_train, labels_train)
#SVC(C=1.0, kernel='linear', gamma=0.0)
print("training time:", round(time()-t0, 3), "s")

t1 = time() #timining time to predict
pred = clf.predict(features_test)
print("prediction time:", round(time()-t1, 3), "s")

accuracy = accuracy_score(labels_test, pred)

print("accuracy:", str(accuracy))
