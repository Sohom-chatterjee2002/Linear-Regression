# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 19:46:15 2021

@author: Sohom chatterjee
"""

#Machine Learning - Linear Regression with Multiple Variable
#Question What is the risk for given data?
#speed=160Km/h, car_age=10 years, experience=5 years

#Import library
import pandas as pd
import numpy as np
from sklearn import linear_model

#Read the dataset using pandas library
df=pd.read_csv('risk_analysis.csv')

#print(df)
mean_value=df.experience.mean()
median_value=df.experience.median()

#print(mean_value)
#print(median_value)
#fill NULL value with median value
df.experience=df.experience.fillna(median_value)
#print(df.experience)
#print(df)

reg=linear_model.LinearRegression()   #Object created for linear regression
reg.fit(df[['speed','car_age','experience']],df.risk)

result=reg.predict([[160,10,5]])  #risk prediction
print(result)

#Printing coefficients and intercept
coefficient=reg.coef_
print("Coefficient value are respectively: ")
print(coefficient)

intercept=reg.intercept_
print("Intercept value is: ")
print(intercept)

#Score calculating
accuracy=reg.score(df[['speed','car_age','experience']],df.risk)
print("Accuracy of model is: ")
print(accuracy*100,'%')
