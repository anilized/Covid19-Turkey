#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 21:37:10 2020

@author: anilcan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

df = pd.read_csv("covid19-in-turkey/covid_19_data_tr.csv")
df_intubated = pd.read_csv("covid19-in-turkey/time_series_covid_19_intubated_tr.csv")
df_intensivecare = pd.read_csv("covid19-in-turkey/time_series_covid_19_intensive_care_tr.csv")

test_num = [1252,1252,1252,1252,1252,1252,1252,1252,1981,5637,8590,10328,14000,17952,22987,
            30273,37806,45447,55429,66964,82386,96782,115539,131699,151363,171428,192828,212851,237751,
            266329,297163,330363,366083,400539,433609,467699,508126,548396,588916,624260,663963,703392,740927,781889, 830257]

print(sum(test_num))

df['Test_Num'] = test_num

print(df['Test_Num'])

print(df.head(10))

df_intubated = df_intubated.T
df_intubated = df_intubated[4:]

ls = df_intubated[0].tolist()

df['Intubated_Count'] = ls

df_intensivecare = df_intensivecare.T
df_intensivecare = df_intensivecare[4:]

ls2 = df_intensivecare[0].tolist()

df['Intensive_Care'] = ls2

# drop province state

sel_cols=['Last_Update', 'Country/Region', 'Confirmed', 'Deaths', 'Recovered', 'Test_Num', 'Intubated_Count', 'Intensive_Care']

df_use = df[sel_cols]
df_use['Last_Update']= pd.to_datetime(df_use['Last_Update'])

df_use['Day'] = df_use['Last_Update'].dt.day
for i in range(0,45):
    df_use['Day'][i] = i
df_use['Month'] = df_use['Last_Update'].dt.month

sel_cols=['Month','Day', 'Country/Region', 'Confirmed', 'Deaths', 'Recovered', 'Test_Num', 'Intubated_Count', 'Intensive_Care']

df_use = df_use[sel_cols]

df_use.to_csv('covidTurkey.csv')