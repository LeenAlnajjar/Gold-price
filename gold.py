# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 19:57:13 2022

Data pre_processing and training of gold data

created by leen alnajjar

"""
#import the important libraries:
import numpy as np
import pandas as pd
import sklearn.metrics as ms
#import the data from csv file:
#hence there are 81 columns use mangle_dupe_cols in case of col duplicated name:
data=pd.read_csv('gold.csv')
#data=pd.read_csv('gold.csv',index_col='Date',parse_dates=True,infer_datetime_format=True)

print(data.info())

#The results show that: 
#1.the number of rows are 1718 
#2.no null values
#3.one column has object (mixed data)
#4.no categrophyical data need for encoding 
#5.no useless columns or rows initially
#6. no duplicated columns name

#check for duplicted values:
print(data.duplicated()) #no duplicated rows

#now check the nununique values:
    
unique=[col for col in data.columns if data[col].nunique()<10] #8 columns have only two values

#Now deal with the date:
from datetime import datetime
#Change the format
#conver date to date:

data["Date"]= pd.to_datetime(data["Date"])

#convert date to numeric:
    
data["Date"]= pd.to_numeric(data["Date"])


#correlation to target values:
corr_matrix=data.corr()
coef=corr_matrix["USO_Adj Close"].sort_values(ascending=False)
#correlation to positive values:
pos_corr=coef[coef>0]
#correlation to negative values:
neg_corr=coef[coef<0]

#Split the data:
X=data.iloc[:,:-1].values #Independet values
y=data.iloc[:,-1].values #Dependent values
'''
#scale the x values:
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
standardscalar = StandardScaler()
X = standardscalar.fit_transform(X[:-1])
y=y.reshape(-1, 1)
y = standardscalar.fit_transform(y)
'''
#Test and training 
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)

print("R2 score for LR =", round(ms.r2_score(y_test,y_pred), 2)) 
X=np.append(arr=np.ones((1718,1)),values=X,axis=1)

import statsmodels.api as sm

def reg_ols(X,y):
    columns=list(range(X.shape[1]))#it will do it just one time
    a={}
    for i in range(X.shape[1]):
        X_opt=np.array(X[:,columns],dtype=float) #every time X_opt will change depend on the columns
        regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()#regressor between x_opt and y
        pvalues = list(regressor_ols.pvalues)#save pvalues column as a list
        d=max(pvalues)#find the maximum value in the list
        if (d>0.06):#if the maximum value bigger than 0.09 then we need to drop the column
            for k in range(len(pvalues)):#for loop to check the values in pvalues to find where is the max
                if(pvalues[k] == d):#if the value of in the index = max the delete it
                    a[k]=d
                    del(columns[k])  
    
    return(X_opt,regressor_ols,a)

X_opt,regressor_ols,a=reg_ols(X, y)
regressor_ols.summary()

#splitting the dataset
from sklearn.model_selection import train_test_split
X_opt_train,X_opt_test,y_opt_train,y_opt_test=train_test_split(X_opt,y,test_size=0.2,random_state=0)

#Training the Simple Linear Reg Model on the Training set
from sklearn.linear_model import LinearRegression
linearRegression2=LinearRegression()
linearRegression2.fit(X_opt_train,y_opt_train)

y_opt_pred=linearRegression2.predict(X_opt_test)

print("R2 score for MR =", (ms.r2_score(y_opt_test,y_opt_pred))) 

from sklearn.ensemble import RandomForestRegressor
regressor = (RandomForestRegressor(n_estimators = 40, random_state = 0))
regressor.fit(X_train, y_train)
y_pred_RF=regressor.predict(X_test)
print('Accuracy of RandomForestRegressor= ',ms.r2_score(y_test, y_pred_RF))

from sklearn.tree import DecisionTreeRegressor
DTR = (DecisionTreeRegressor(random_state = 0))
DTR.fit(X_train, y_train)
y_pred_DTR=DTR.predict(X_test)
print('Accuracy of DecisionTreeRegressor= ',ms.r2_score(y_test, y_pred_DTR))
#Try the Ridge algorithm:
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
standardscaler=StandardScaler()
X_train_sc=standardscaler.fit_transform(X_train)
X_test_sc=standardscaler.fit_transform(X_test)  

ridge=Ridge()
ridge.fit(X_train_sc,y_train)
y_pred_ridge=ridge.predict(X_test_sc)
print('Accuracy of Ridge= ',ms.r2_score(y_test, y_pred_ridge))

# Support Vector regressor algorithm:
from sklearn.svm import SVR
SV=(SVR(kernel = 'linear')) #in this case linear has the highest accuracy
SV.fit(X_train,y_train)
y_pred_SV=SV.predict(X_test_sc)
print('Accuracy of SVR= ',ms.r2_score(y_test, y_pred_SV)) 



#Results of comparision:
'''
R2 score for MR = 0.95
Accuracy of RandomForestRegressor=  0.9999724303969466
Accuracy of DecisionTreeRegressor=  0.9997286917432303
Accuracy of Ridge=  0.9976570343123081
'''

'''
Ridge regression is a term used to refer to a linear regression model whose coefficients are estimated not by ordinary least squares (OLS), but by an estimator, called ridge estimator, that, albeit biased, has lower variance than the OLS estimator.

In certain cases, the mean squared error of the ridge estimator (which is the sum of its variance and the square of its bias) is smaller than that of the OLS estimator.
'''