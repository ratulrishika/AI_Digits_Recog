# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 10:48:52 2020

@author: Ratul
"""

import sklearn

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
data=load_digits()
print(data.target_names)
print(data.target)
print(data.target.shape)
print(data.data[0])
xx=data.images[0]
plt.imshow(xx)
plt.show(xx)


def imageshow_des(p):
    print(data.target[p])
    print(data.data[p])
    print(data.data[p].reshape(8,8))
    
def imageshow_im(p):
    xx=data.images[p]
    plt.imshow(xx)
    
imageshow_des(893)
imageshow_im(893)

from sklearn.utils import shuffle
import numpy as np
import sklearn.svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,gridsearchcv,randomgridsearcv
from sklearn.metrics import confusion_matrix,classification_report


x_train,x_test,y_train,y_test=train_test_split(data.data,data.target,test_size=0.2)
print(x_train.shape)
print(y_train.shape)

model=SVC(C=1,gamma=0.001,kernel='rbf')
model.fit(x_train,y_train)
print(model.score(x_train,y_train))
print(model.score(x_test,y_test))
y_pred=model.predict(x_test)
print(confusion_matrix(y_pred,y_test))
print(classification_report(y_pred,y_test))

print(y_test[30])
print(y_pred[30])
