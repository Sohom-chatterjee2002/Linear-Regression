# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 15:14:21 2021

@author: Sohom Chatterjee
"""

#Machine Learning - Linear Regression
#Car driving risk analysis using Linear Regression model

#importing library
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read the CSV file
df=pd.read_csv('car_driving_risk_analysis.csv')

#Separating Variable
x=df[['speed']]     #Speed is independent variable
y=df[['risk']]      #Risk is dependent variable which depends on speed

#separation of training and testing data
from sklearn.model_selection import train_test_split   
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.40,random_state=1)


#Linear Regression
from sklearn.linear_model import LinearRegression
reg=LinearRegression()   # Object Creted for Linear Regression

reg.fit(xtrain,ytrain)   #For training purpose

result=reg.predict(xtest)  #For prediction purpose

#Prediction result for testing data
print(xtest)
print(ytest)
print(result)

#Result and Algorithm visualization
plt.scatter(df['speed'],df['risk'],marker='*',color='red')
plt.title('Car Driving Risk Analysis')
plt.xlabel('Speed in Km/h')
plt.ylabel('Risk in number of cases')
plt.plot(df.speed,reg.predict(df[['speed']]))
plt.show()

#Printing Intercept and coefficient
intercept=reg.intercept_
print("Intercept: ",intercept)

coefficient=reg.coef_
print("Coefficient: ",coefficient)

#Predicting for a random value
res=reg.predict([[296]])
print("Risk calculated for the given value is: ")
print(res)

#Accuracy of model
accuracy=reg.score(xtest,ytest)
print("Accuracy of model is: ")
print(accuracy*100,'%')