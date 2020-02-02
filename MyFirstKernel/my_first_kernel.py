# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 23:35:12 2020

@author: Deniz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("heart.csv")
data.head()

data.info()

x = np.array(data.loc[:,'age']).reshape(-1,1)
y = np.array(data.loc[:,'thalach']).reshape(-1,1)

# Scatter
plt.figure(figsize=[10,10])
plt.scatter(x=x,y=y,color="red")
plt.xlabel("Age")       # name of label
plt.ylabel("Maximum Heart Rate")
plt.title("MAXIMUM HEART RATE BY AGES")      # title of plot
plt.show()

#%%

from sklearn.linear_model import LinearRegression    #sklearn library

linear_reg=LinearRegression()     #linear regression model

predict_space = np.linspace(min(x), max(x)).reshape(-1,1)
linear_reg.fit(x,y)
predicted = linear_reg.predict(predict_space)

print('R^2 score: ',linear_reg.score(x, y))
# Plot regression line and scatter
plt.plot(predict_space, predicted,color='black', linewidth=3)
plt.scatter(x=x,y=y,color="red")
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()
