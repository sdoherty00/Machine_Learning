# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:10:55 2022

@author: campu
"""

import IPython as IP 
IP.get_ipython().magic('reset -sf')

import numpy as np
import scipy as sp
import pandas as pd
from scipy import fftpack, signal # have to add 
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import linear_model


def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", 
                              values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]
   
plt.close('all')

#%% Load and plot data
oecd_bli = pd.read_csv("data/oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("data/gdp_per_capita.csv",thousands=',',
                             delimiter='\t', encoding='latin1', na_values="n/a")

# Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
GDP = country_stats.values[:,0]
life_satisfaction = country_stats.values[:,1]
countries = list(country_stats.index)

# Normalize?
cleaned_GDP = np.delete(GDP,GDP==50000)
cleaned_life_satisfaction = np.delete(life_satisfaction, GDP==50000)

plt.figure()
plt.plot(cleaned_GDP,cleaned_life_satisfaction,'o')
plt.xlabel('Average Income')
plt.ylabel('How Happy People are')

X = cleaned_GDP
Y = cleaned_life_satisfaction


#%% Build a model

GDP_model_X = np.linspace(0,60000)

theta_1 = 4.7
theta_2 = (1/20000)
life_satisfaction_model_Y = theta_1 + theta_2 * GDP_model_X


plt.figure()
plt.plot(X,Y,'o')
plt.plot(GDP_model_X,life_satisfaction_model_Y)
plt.xlabel('Average number of rooms per dwelling')
plt.ylabel("Median value of owner-occupied homes in $1000's")
plt.grid(True)
plt.xlim([0,60000])
plt.ylim([4.5,8])
plt.legend(framealpha = 1)
plt.tight_layout()

# add a dimension to the data as much is easier
# in 2d arrays
X = np.expand_dims(X, axis=1)
Y = np.expand_dims(Y, axis=1)
GDP_model_X = np.expand_dims(GDP_model_X, axis=1)

#compute the linear regression solution using
X_b = np.ones((X.shape[0], 2))
X_b[:,1] = X.T # add x0=1 to each instance

theta_closed_form = np.linalg.inv(X_b.T@X_b)@X_b.T.dot(Y)
life_satisfaction_model_Y_closed_form = theta_closed_form[0] + theta_closed_form[1] * GDP_model_X

plt.figure()
plt.plot(X,Y,'o',label='data')
plt.plot(GDP_model_X,life_satisfaction_model_Y_closed_form, '--', label='closed form')
plt.xlabel('Average number of rooms per dwelling')
plt.ylabel("Median value of owner-occupied homes in $1000's")
plt.grid(True)
plt.xlim([0,60000])
plt.ylim([4.5,8])
plt.legend(framealpha = 1)
plt.tight_layout()

# compute the linear regression solution using gradient descent

eta = 0.004 # learning rate
n_iterations = 100
m = X.shape[0]
theta_gradient_descent = np.random.randn(2,1) # randon initialization

for iteration in range(n_iterations):
    gradient = 2/m*X_b.T.dot(X_b.dot(theta_gradient_descent) - Y)
    theta_gradient_descent = theta_gradient_descent - eta*gradient
   
print(theta_gradient_descent)

life_satisfaction_model_y_gradient_descent = theta_gradient_descent[0] + theta_gradient_descent[1] * GDP_model_X

plt.figure()
plt.title('Iterations')
plt.plot(X,Y,'o',label='data')
plt.plot(GDP_model_X, life_satisfaction_model_Y_closed_form, '--', label='closed form')
plt.plot(GDP_model_X,life_satisfaction_model_y_gradient_descent, label='gradient descent')
plt.xlabel('Average number of rooms per dwelling')
plt.ylabel("Median value of owner-occupied homes in $1000's")
plt.grid(True)
plt.xlim([0,60000])
plt.ylim([4.5,8])
plt.legend(framealpha = 1)
plt.tight_layout()

#%% Compute the linear regression solution using SK learn

model = sk.linear_model.LinearRegression()
model.fit(X,Y)
life_satisfaction_model_y_sklearn = model.predict(GDP_model_X)

plt.figure()
plt.plot(X,Y,'o',label='data')
plt.plot(GDP_model_X,life_satisfaction_model_Y, label='manual fit')
plt.plot(GDP_model_X, life_satisfaction_model_Y_closed_form, '--', label='closed form')
plt.plot(GDP_model_X,life_satisfaction_model_y_gradient_descent, label='gradient descent')
plt.plot(GDP_model_X,life_satisfaction_model_y_sklearn, ':', label='sk learn')
plt.xlabel('Average number of rooms per dwelling')
plt.ylabel("Median value of owner-occupied homes in $1000's")
plt.grid(True)
plt.xlim([0,60000])
plt.ylim([4.5,8])
plt.legend(framealpha = 1)
plt.tight_layout()
