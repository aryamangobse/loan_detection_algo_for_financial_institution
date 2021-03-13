# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



import pandas as pd
import numpy as np
train_data=pd.read_csv(r'C:\\Users\AryamanGobse\OneDrive\projectarya\risk_analytics_train (1).csv',header=0)
test_data=pd.read_csv(r'C:\\Users\AryamanGobse\OneDrive\projectarya\risk_analytics_test (1).csv',header=0)

print(train_data.shape)
train_data.head()

print(train_data.isnull().sum())

colname1=["Gender","Married","Dependents","Self_Employed","Loan_Amount_Term"]
for x in colname1:
    train_data[x].fillna(train_data[x].mode()[0],inplace=True)

print(train_data.isnull().sum())

train_data["LoanAmount"].fillna(train_data["LoanAmount"].mean(),inplace=True)
print(train_data.isnull().sum())

train_data["Credit_History"].fillna(value=0, inplace=True)
print(train_data.isnull().sum())

from sklearn import preprocessing
colname=["Gender","Married","Education","Self_Employed","Property_Area","Loan_Status"]

le=preprocessing.LabelEncoder()

for x in colname:
    train_data[x]=le.fit_transform(train_data[x])

print(test_data.shape)
test_data.head()

print(test_data.isnull().sum())

colname1=["Gender","Dependents","Self_Employed","Loan_Amount_Term"]
for x in colname1:
    test_data[x].fillna(test_data[x].mode()[0],inplace=True)

print(test_data.isnull().sum())

test_data["LoanAmount"].fillna(test_data["LoanAmount"].mean(),inplace=True)
print(test_data.isnull().sum())

test_data["Credit_History"].fillna(value=0, inplace=True)
print(test_data.isnull().sum())

from sklearn import preprocessing
colname=["Gender","Married","Education","Self_Employed","Property_Area"]

le=preprocessing.LabelEncoder()

for x in colname:
    test_data[x]=le.fit_transform(test_data[x])

test_data.dtypes
test_data.head()

X_train=train_data.values[:,1:-1]
Y_train=train_data.values[:,-1]
Y_train=Y_train.astype(int)

X_test=test_data.values[:,1:]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn import svm
svc_model = svm.SVC(kernel='rbf',C=1.0,gamma=0.1)

svc_model.fit(X_train,Y_train)
Y_pred=svc_model.predict(X_test)
print(list(Y_pred))

Y_pred_col = list(Y_pred)

test_data=pd.read_csv(r'C:\\Users\AryamanGobse\OneDrive\projectarya\risk_analytics_test (1).csv',header=0)
test_data["Y_predictions"]=Y_pred_col
test_data.head()

test_data.to_csv('test_data.csv')

classifier=svm.SVC(kernel='rbf',C=10.0,gamma=0.001)

from sklearn.model_selection import KFold
kfold_cv=KFold(n_splits=10)
print(kfold_cv)
from sklearn.model_selection import cross_val_score

kfold_cv_result=cross_val_score(estimator=classifier,X=X_train,
y=Y_train, cv=kfold_cv)
print(kfold_cv_result)

print(kfold_cv_result)

print(kfold_cv_result.mean())

import pickle
pickle_out = open("classifier.pkl", "wb")
pickle.dump(svc_model, pickle_out)
pickle_out.close()

