import numpy as np
import pandas as pd

df = df = pd.read_csv("/Users/anilcan/Desktop/BAU/Data Science/PyFiles/LinearReg/covid-homework/covidTurkey.csv")
df = df.drop("Unnamed: 0", axis=1)

x = df.iloc[:, [3,5,6,7,8]].values
y = df.iloc[:, 4].values

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
        