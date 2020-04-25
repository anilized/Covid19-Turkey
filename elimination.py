import numpy as np
import pandas as pd

df = df = pd.read_csv("/Users/anilcan/Desktop/BAU/Data Science/PyFiles/LinearReg/covid-homework/covidTurkey.csv")


# BACKWARD ELIMINATION

x = df.iloc[:, [3,5,6,7,8]].values
y = df.iloc[:, 4].values

from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

print(regressor.score(x_test, y_test)) # # 0.9991942639750137

import statsmodels.api as sm
x = np.append(arr = np.ones((45,1)).astype(float), values = x, axis = 1)
x_opt = x[:, [0,1,2,3,4,5]]
regressor_ols = sm.OLS(y, x_opt).fit()
#print(regressor_ols.summary())

s1 = 0.05

num_vars = len(x_opt[0])
for i in range(0, num_vars):
    regressor_ols = sm.OLS(y, x_opt).fit()
    print("i: ", i)
    print(regressor_ols.summary())
    maxVar = max(regressor_ols.pvalues).astype(float)
    if maxVar > s1:
        for j in range(0, num_vars - i):
            if(regressor_ols.pvalues[j].astype(float) == maxVar):
                x_opt = np.delete(x_opt, j, 1)

# choices are x1,x2,x3       
                
x1 = df.iloc[:, [3,5,6]].values

from sklearn.model_selection import train_test_split 
x1_train, x1_test, y_train, y_test = train_test_split(x1, y, test_size = 1/3, random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x1_train, y_train)

print(regressor.score(x1_test, y_test)) # 0.9992122709590304

