#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 22:33:12 2020

@author: anilcan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/Users/anilcan/Desktop/BAU/Data Science/PyFiles/LinearReg/covid-homework/covidTurkey.csv")

"""
data.corr()
f, ax = plt.subplots(figsize=(7,7))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# Line Plot
data.Test_Num.plot(kind = 'line', color = 'g',label = 'Test_Num',linewidth=1,alpha = 1,grid = True,linestyle = ':')
data.Confirmed.plot(color = 'r',label = 'Confirmed',linewidth=1, alpha = 1,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('Day')              # label = name of label
plt.ylabel('Total number')
plt.title('Line Plot')            # title = title of plot
plt.show()


# Scatter Plot 

data.plot(kind='scatter', x='Test_Num', y='Confirmed',alpha = 0.5,color = 'red')
plt.xlabel('Test_Num')             
plt.ylabel('Confirmed')
plt.title('Test Num - Confirmed Scatter Plot')     
plt.show()       



# Bar Plot
plt.figure(figsize=(15,10))
sns.barplot(x=data['Day'], y=data['Deaths'])
plt.xticks(rotation= 45)
plt.xlabel('Day')
plt.ylabel('Deaths')
plt.title('Total Deaths')


plt.figure(figsize=(15,10))
sns.barplot(x=data['Day'], y=data['Confirmed'])
plt.xticks(rotation= 45)
plt.xlabel('Day')
plt.ylabel('Confirmed')
plt.title('Total Confirmed')

"""

"""
x = df['Day'].values #get column Live Area
y = df['Confirmed'].values #get column Sale Price

x = x.reshape(-1, 1) #independent
y = y.reshape(-1, 1) #dependent

#split dataset into train and test splits
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

#fit simple linear regression model on training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

#training set results
plt.figure()
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Days vs Confirmed Cases (training set)')
plt.xlabel('Days')
plt.ylabel('Confirmed Cases')
plt.show()

#test set results
plt.figure()
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Days vs Confirmed Cases (test set)')
plt.xlabel('Days')
plt.ylabel('Confirmed Cases')
plt.show()

msqe = sum((y_pred - y_test) * (y_pred - y_test)) / y.shape[0]
rmse = np.sqrt(msqe)
print(msqe, rmse)   

msqe[40644262.06205162] rmse[6375.28525339]
"""

"""
sel_cols = ['Deaths', 'Month', 'Day', 'Recovered', 'Confirmed', 'Test_Num', 'Intubated_Count', 'Intensive_Care']
df_multi = df[sel_cols]

x = df_multi.iloc[:, 1:].values
y = df_multi.iloc[: , 0].values


from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)


#fit multiple linear regression model on training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

msqe = sum((y_pred - y_test) * (y_pred - y_test)) / y.shape[0]
rmse = np.sqrt(msqe)

print(msqe, rmse)

# 168.20317286822623 12.969316592181187

"""

"""
x = df['Confirmed'].values 
y = df['Deaths'].values 

x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

y_pred = lin_reg.predict(x)

#visualize simple linear regression
plt.figure()
plt.scatter(x, y, color = 'red')
plt.plot(x, y_pred, color = 'blue')
plt.title('Confirmed vs Deaths')
plt.xlabel('Confirmed Cases')
plt.ylabel('Deaths')
plt.show()

#finding error
msqe = sum((y_pred - y) * (y_pred - y)) / y.shape[0]
rmse = np.sqrt(msqe)

#polynomial linear regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)

#sort x
    
s_x = np.sort(x, axis=None).reshape(-1, 1)

#visualize polynomial linear regression
plt.scatter(x, y, color = 'red')
plt.plot(s_x, lin_reg2.predict(poly_reg.fit_transform(s_x)), color = 'blue')
plt.title('Confirmed vs Deaths')
plt.xlabel('Confirmed Cases')
plt.ylabel('Deaths')
plt.show()
"""


sel_cols = ['Deaths', 'Month', 'Day', 'Recovered', 'Confirmed', 'Test_Num', 'Intubated_Count', 'Intensive_Care']
df_multi = df[sel_cols]
df_multi = df_multi.loc[17:27]

x = df_multi['Intensive_Care'].values
y = df_multi['Intubated_Count'].values

x = x.reshape(-1,1)
y = y.reshape(-1,1)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0, max_depth=2) # for max_depth >= 3 it overfits
regressor.fit(x,y)

y_pred = regressor.predict(x)

s_x = np.sort(x, axis=None).reshape(-1,1)

plt.figure()
x_grid = np.arange(min(s_x),max(s_x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(x,y,color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue' )
plt.title("Intensive Care vs Intubated Count")
plt.xlabel("Intensive Care")
plt.ylabel("Intubated Count")
plt.show()

msqe = sum((y_pred - y) * (y_pred - y)) / y.shape[0]
rmse = np.sqrt(msqe)
print(msqe, rmse)
"""
"""

"""
sel_cols = ['Deaths', 'Month', 'Day', 'Recovered', 'Confirmed', 'Test_Num', 'Intubated_Count', 'Intensive_Care']
df_multi = df[sel_cols]

x = df_multi.iloc[:, 3:].values
y = df_multi.iloc[: , 0].values


from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)


#fit multiple linear regression model on training set
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0, max_depth=2)
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

msqe = sum((y_pred - y_test) * (y_pred - y_test)) / y.shape[0]
rmse = np.sqrt(msqe)

print(msqe, rmse) -> 23845.479124999998 154.41981454787464
"""
"""

# Random Forest Regression

x = df.iloc[1:30,1].values.reshape(-1,1)
y = df.iloc[1:30,3].values.reshape(-1,1)


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(x,y)


x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = rf.predict(x_)

# visualize
plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color="green")
plt.xlabel("Number of days")
plt.ylabel("Number of cases")
plt.show()

"""







