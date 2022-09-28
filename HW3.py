# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 12:29:40 2022

@author: campu
"""

import IPython as IP
IP.get_ipython().magic('reset -sf')

import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import linear_model
from sklearn import pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

plt.close('all')

#%% Import Data

X = np.loadtxt('X.txt')
Y = np.loadtxt('Y.txt')
X_train = np.loadtxt('x_train.txt')
X_val = np.loadtxt('x_validation.txt')
Y_train = np.loadtxt('y_train.txt')
Y_val = np.loadtxt('y_validation.txt')


X_train = np.expand_dims(X_train, axis=1)
X_val = np.expand_dims(X_val, axis=1)
Y_train = np.expand_dims(Y_train, axis=1)
Y_val = np.expand_dims(Y_val, axis=1)

# plt.scatter(X,Y)

#%% Build Model

train_poly_errors, val_poly_errors = [], []

poly = PolynomialFeatures(degree=20,include_bias=False)

poly_features = poly.fit_transform(X_train)

poly_reg_model = LinearRegression()
poly_reg_model.fit(poly_features,Y_train)

y_predicted = poly_reg_model.predict(poly_features)

plt.plot(np.linspace(X_train.min(),X_train.max(),len(Y_train)),y_predicted)

# plt.figure()
# plt.scatter(X,Y)
# plt.plot(np.linspace(X.min(),X.max(),32),y_predicted)


for i in range(1,len(X_train)+1):
    model = sk.pipeline.make_pipeline(sk.preprocessing.PolynomialFeatures(20),sk.linear_model.Ridge(alpha=1))
    model.fit(X_train[:i], Y_train[:i])
    Y_train_predict = model.predict(X_train[:i])
    Y_val_predict = model.predict(X_val)
   
    # compute the error for the train model
    mse_train = sk.metrics.mean_absolute_error(Y_train[:i],Y_train_predict)
    train_poly_errors.append(mse_train)
   
    # compute the error for the val model
    mse_val = sk.metrics.mean_absolute_error(Y_val,Y_val_predict)
    val_poly_errors.append(mse_val)
   
    plt.figure()
    plt.scatter(X,Y,s=2,label='data')
    plt.scatter(X_train[:1],Y_train[:1], s=2, label='training data')
    plt.scatter(X_val, Y_val, marker='s', label='validation data')
    Y_model=model.predict(np.expand_dims(X,axis=1))
    plt.plot(X,Y, 'r--', label='model')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=2)
    plt.grid(True)
    plt.close()
   
   
plt.figure()
plt.grid(True)
plt.plot(train_poly_errors,'--',label='train')
plt.plot(val_poly_errors, '--', label='val')
plt.xlabel('number of data points in train set')
plt.ylabel('mean squared error')
plt.legend()

plt.figure()
plt.scatter(X,Y,s=2,label='data')
plt.scatter(X_train,Y_train, label='training data')
plt.scatter(X_val, Y_val, marker='s', label='validation data')
Y_model=model.predict(np.expand_dims(X,axis=1))
plt.plot(np.linspace(X.min(),X.max(),40),Y_model, 'r--', label='Ridge model')
plt.plot(np.linspace(X_train.min(),X_train.max(),len(Y_train)),y_predicted,label = 'Poly Model')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(0,14)
plt.legend(loc=2)
plt.grid(True)



















































































































