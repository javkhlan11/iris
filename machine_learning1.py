#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 21:35:14 2019

@author: javkhlanenkhbold
"""
import pandas as pd
import numpy as np 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 



iris_datasets = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_datasets["data"], 
                                                    iris_datasets["target"], 
                                                    random_state = 0)

iris_DataFrame = pd.DataFrame(X_train, columns = iris_datasets.feature_names) 
grr = pd.plotting.scatter_matrix(iris_DataFrame, c = y_train, figsize= (15,15), 
                                 hist_kwds = {"bins":20}, marker = "o", s= 60 , alpha =.8 )

from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors =1)
knn.fit(X_train, y_train)

#Found new IRIS !

X_new = np.array([[1, 3, 5, .2]])

prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("predicted Name : {}".format(iris_datasets["target_names"][prediction]))

# Predicting all X_test 
y_pred = knn.predict(X_test)
print("Vorhersagen für den Testdatensatz : {}".format(y_pred))

# Accuracy of our model 
print("accuracy of test data : {:.2f}".format(np.mean(y_pred == y_test)))
#or
print("accuracy of test data : {:.2f}".format(knn.score(X_test, y_test)))


