# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:28:30 2023

@author: vivek
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from pickle import dump
from pickle import load
from sklearn.metrics import matthews_corrcoef

datasets = pd.read_csv('C:/Users/sony/Desktop/Files_for_test_Multilingual/GFCC_Vowel_a_German.csv')
X = datasets.iloc[:, 0:13].values
y = datasets.iloc[:, 13].values

X_train, X_test, y_train, y_test = train_test_split(X, y)

sc = StandardScaler()
X = sc.fit_transform(X)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)

param_grid = {'n_estimators': [50,100,200,300,500,700,1000], 'max_features': ['auto', 'sqrt', 'log2'], 'max_depth' : [4,5,6,7,8], 'criterion' :['entropy', 'gini']}
model = RandomForestClassifier ()

loocv = LeaveOneOut()
scoring = 'accuracy'
kfold = KFold(n_splits=10, random_state=None)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

best_model = grid_result.best_estimator_
best_model.fit(X_train, y_train)
filename = 'GFCC_German_A_RF1.sav'
dump(best_model,open(filename, 'wb'))
loaded_model = load(open(filename, 'rb'))

y_pred1 = loaded_model.predict(X_test)
MCC=matthews_corrcoef(y_test, y_pred1)

cm1 = confusion_matrix(y_test, y_pred1)
acc = (cm1[0,0]+cm1[1,1])/(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1])
spec = (cm1[0,0])/(cm1[0,0]+cm1[0,1])
sens = (cm1[1,1])/(cm1[1,0]+cm1[1,1])

print('Testing Accuracy =' ,acc)
print('Testing Sensitivity(abnormality) =' ,sens)
print('Testing Specificity(normality) =' ,spec)
print ('confusion matrix =', cm1)
print ('MCC Score =', MCC)
print(X.shape, y.shape)
