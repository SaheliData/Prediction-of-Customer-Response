#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: saheli06
"""


import pandas as pd

dataset = pd.read_csv("marketing_training.csv")
cleaned_data = dataset.copy() #backup of the original file

# Deleting the observations where profession variable has unknown values. There are sixty one observations deleted.
cleaned_PCA_data = cleaned_PCA_data.drop(cleaned_PCA_data[cleaned_PCA_data.profession == 'unknown'].index)
dataf = cleaned_PCA_data.copy()

# unknown values for features - 'schooling, 'marital', 'default', 'housing', 'loan', 'day_of_week' are converted to the most frequent 
# occurred values. If there are only two kind of value like (yes/no) then they are converted to 0 and 1. 

# Features like 'pdays', 'pmonths' has 999 value for client was not previously contacted. My machine model can treat this high value with high importance
# so I converted it to 0 values as the client was not previously contacted.
dataf = dataf.replace({'schooling': {'unknown': 'university.degree'},
                       'marital': {'unknown': 'married' },
                        'default': {'yes': 1, 
                                   'no': 0,
                                   'unknown': 0},
                         'housing': {'yes': 1, 
                                     'no': 0,
                                     'unknown': 1},
                         'loan': {'yes': 1, 
                                            'no': 0,
                                            'unknown': 0
                                            },
                        'contact': {'cellular': 0, 
                                         'telephone': 1},
                        'responded': {'yes': 1, 
                                            'no': 0},
                        'day_of_week': {'unknown': 'mon'},
                                'pdays':{999 : 0},
                               'pmonths':{999 : 0} 
                        })

# code for converting categorical variables into Dummy Variables
dataf_prof = pd.get_dummies(dataf['profession'])
dataf_schooling = pd.get_dummies(dataf['schooling'])
dataf_marital = pd.get_dummies(dataf['marital'])
dataf_month = pd.get_dummies(dataf['month'])
dataf_day_of_week = pd.get_dummies(dataf['day_of_week'])
dataf_poutcome = pd.get_dummies(dataf['poutcome'])

# After conversion deleting the features with original or categorical values
del dataf['profession']
del dataf['schooling']
del dataf['marital']
del dataf['month']
del dataf['day_of_week']
del dataf['poutcome']

# Preparation of the independent features
dataf = pd.concat([dataf, dataf_prof], axis = 1)
dataf = pd.concat([dataf, dataf_schooling], axis = 1)
dataf = pd.concat([dataf, dataf_marital], axis = 1)
dataf = pd.concat([dataf, dataf_month], axis = 1)
dataf = pd.concat([dataf, dataf_day_of_week], axis = 1)
dataf = pd.concat([dataf, dataf_day_of_week], axis = 1)

# 'custAge' is the only one quantitative variable with missing values. These are replaced by median of the column.
# I checked with mean also getting same result.
dataf['custAge'] = dataf['custAge'].fillna(int(dataf.custAge.median()))
X1 = dataf.iloc[:, 0:15].values
X1 = pd.DataFrame(X1)
X2 = dataf.iloc[:, 16:57].values
X2 = pd.DataFrame(X2)         
X1 = pd.concat([X1, X2], axis = 1)                                   
y = dataf.iloc[:, 21].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size = 0.20, random_state = 0)
                           
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Using Factor Analysis for feature selection
from sklearn.decomposition import FactorAnalysis
fca = FactorAnalysis(n_components = 16)
X_train = fca.fit_transform(X_train)
X_test = fca.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C = 1, random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_lg = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_lg = confusion_matrix(y_test, y_pred_lg)

# Checking Accuracy score
classifier.score(X_test, y_test)

# Checking Classification report
from sklearn.metrics import classification_report
report_lg = classification_report(y_test, y_pred_lg)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Applying Grid Search to find the best model and the best parameters
#from sklearn.model_selection import GridSearchCV
#parameters = [{'C': [1, 10, 100, 1000]}]
#grid_search = GridSearchCV(estimator = classifier,
#                           param_grid = parameters,
#                           scoring = 'accuracy',
#                           cv = 10,
#                           n_jobs = -1)
#grid_search = grid_search.fit(X_train, y_train)
#best_accuracy = grid_search.best_score_
#best_parameters = grid_search.best_params_
