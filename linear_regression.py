# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 12:25:09 2021

@author: Sohom Chatterjee
"""

#Machine Learning - Linear Regression
# Home prices analysis based on area


#importing library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv('dataset.csv')

#print(df.head(3))  #To show 3 data from head of your dataset
#print(df.shape)   #To show shape of your dataset
#print(df.isnull().any())   #To check wheather your dataset is null or not
#print(df.isnull().sum())   #To check how many data is null or not in your dataset


#Separation of dataset
x=df[['area']]   #x= independent variable
y=df[['price']]  #y= dependent variable which depends on x

#Linear Regression equation: y=mx+c

#visualization
#plt.scatter(df['area'],df['price'],marker='*',color='red')
#plt.title('Home Prices')
#plt.xlabel('Area in square ft.')
#plt.ylabel('Price in Ruppes')
#plt.show()


#separation of training and testing data
from sklearn.model_selection import train_test_split   
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.30,random_state=1)
#Printing data for test and train which was randomly choose
#print(xtrain)
#print(xtest)
#print(ytrain)
#print(ytest)


#Linear Regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()   #object created for Linear regression

reg.fit(xtrain,ytrain)    #For training purpose

result=reg.predict(xtest)      #For prediction purpose


print(xtest)
print(ytest)
print(result)

#Result and Algorithm visualization
plt.scatter(df['area'],df['price'],marker='*',color='red')
plt.title('Home Prices')
plt.xlabel('Area in square ft.')
plt.ylabel('Price in Ruppes')
plt.plot(df.area,reg.predict(df[['area']]))
plt.show()

#Prediction for a random value
#print(reg.predict([(3500)])
res=reg.predict([[3900]])
print("Price calculated for the given value of area: ")
print(res)

#Checking coefficient value of model
coefficient=reg.coef_
print(coefficient)

#checking intercept value of model
intercept=reg.intercept_
print(intercept)

#Accuracy of model
accuracy=reg.score(xtest,ytest)
print("Accuracy of model is: ")
print(accuracy*100,'%')
