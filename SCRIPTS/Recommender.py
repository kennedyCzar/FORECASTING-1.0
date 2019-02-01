# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 12:56:41 2018

@author: kennedy
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import (AdaBoostRegressor, #Adaboost regressor
                              RandomForestRegressor, #Random forest regressor
                              GradientBoostingRegressor, #Gradient boosting
                              BaggingRegressor, #Bagging regressor
                              ExtraTreesRegressor) #Extratrees regressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import (KNeighborsRegressor, RadiusNeighborsRegressor)
from sklearn.linear_model import (LinearRegression, #linear rgeressor
                                  LogisticRegression, #logistic regressor
                                  RANSACRegressor, #RANSACR regress
                                  Ridge, #Ridge regressor
                                  Lasso)# Lasso regressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, #accuracy
                             confusion_matrix, #confusion matrix
                             classification_report) #classification report
from sklearn.model_selection import GridSearchCV
from TechnicalIndicators import TechnicalIndicators
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, StandardScaler)
from sklearn.model_selection import cross_validate





class Model():
  def __init__(self):
    '''
    Define the Classifiers to be Used for 
    @Classifiers:
                List of Tuples
    @Pipeline: Channel of Estimators
    @Employ the use of GridSearchCV
    Predicting Returns
    '''
    self.N_NEIGBORS = 10
    self.KERNELS = ['linear', 'rbf']
    self.GAMMA = [0.0001, 0.001, 0.01, 1]
    self.CRITERION = ['gini', 'entropy']
    self.MAX_DEPTH = 5
    self.MAX_FEATURES = ['auto', 'sqrt', 'log2']
    self.N_VALIDATION = 2
    self.N_COMPONENTS = 2
    self.BEST_ACCURACY = 0.0
    self.BEST_CLASSIFIER = 0
    self.BEST_GRIDSEARCH = ''
    
    #Support vector regressor
    self.pipe_SVR = Pipeline([('normalizer', StandardScaler()), ('clf', SVR())])
    self.pipe_SVR_PCA = Pipeline([('normalizer', StandardScaler()), ('PCA', PCA(n_components = self.N_COMPONENTS)),
                          ('clf', SVR())])
    #Adaboost regressor
    self.pipe_AdaBoostRegressor = Pipeline([('normalizer', StandardScaler()), ('clf', AdaBoostRegressor())])
    self.pipe_AdaBoostRegressor_PCA = Pipeline([('normalizer', StandardScaler()), ('PCA', PCA(n_components = self.N_COMPONENTS)), 
                          ('clf', AdaBoostRegressor())])
    #RandomForest Regressor
    self.pipe_RandomForestRegressor = Pipeline([('normalizer', StandardScaler()), 
                                            ('clf', RandomForestRegressor())])
    self.pipe_RandomForestRegressor_PCA = Pipeline([('normalizer', StandardScaler()), ('PCA', PCA(n_components = self.N_COMPONENTS)), 
                                                ('clf', RandomForestRegressor())])
    #Gradient boosting regressor
    self.pipe_GradientBoostingRegressor = Pipeline([('normalizer', StandardScaler()), ('clf', GradientBoostingRegressor())])
    self.pipe_GradientBoostingRegressor_PCA = Pipeline([('normalizer', StandardScaler()),('PCA', PCA(n_components = self.N_COMPONENTS)), 
                                       ('clf', GradientBoostingRegressor())])
    #Bagging regressor
    self.pipe_BaggingRegressor = Pipeline([('normalizer', StandardScaler()), ('clf', BaggingRegressor())])
    self.pipe_BaggingRegressor_PCA = Pipeline([('normalizer', StandardScaler()), ('PCA', PCA(n_components = self.N_COMPONENTS)), 
                                             ('clf', BaggingRegressor())])
    #Extratrees regressor
    self.pipe_ExtraTreesRegressor = Pipeline([('normalizer', StandardScaler()), ('clf', ExtraTreesRegressor())])
    self.pipe_ExtraTreesRegressor_PCA = Pipeline([('normalizer', StandardScaler()), ('PCA', PCA(n_components = self.N_COMPONENTS)), 
                                             ('clf', ExtraTreesRegressor())])
    #DecisionTreeRegressor
    self.pipe_DecisionTreeRegressor = Pipeline([('normalizer', StandardScaler()), 
                                     ('clf', DecisionTreeRegressor())])
    self.pipe_DecisionTreeRegressor_PCA = Pipeline([('normalizer', StandardScaler()), ('PCA', PCA(n_components = self.N_COMPONENTS)), 
                                         ('clf', DecisionTreeRegressor())])
    #KNeighborsRegressor
    self.pipe_KNeighborsRegressor = Pipeline([('normalizer', StandardScaler()), ('clf', KNeighborsRegressor())])
    self.pipe_KNeighborsRegressor_PCA = Pipeline([('normalizer', StandardScaler()), ('PCA', PCA(n_components = self.N_COMPONENTS)), 
                                 ('clf', KNeighborsRegressor())])
    #RadiusNeighborsRegressor
    self.pipe_RadiusNeighborsRegressor = Pipeline([('normalizer', StandardScaler()), 
                                                ('clf', RadiusNeighborsRegressor())])
    self.pipe_RadiusNeighborsRegressor_PCA = Pipeline([('normalizer', StandardScaler()), 
                                                ('PCA', PCA(n_components = self.N_COMPONENTS)), ('clf', RadiusNeighborsRegressor())])
    #LinearRegression
    self.pipe_LinearRegression = Pipeline([('normalizer', StandardScaler()), 
                                            ('clf', LinearRegression())])
    self.pipe_LinearRegression_PCA = Pipeline([('normalizer', StandardScaler()), 
                                                ('PCA', PCA(n_components = self.N_COMPONENTS)), ('clf', LinearRegression())])
  
    #LogisticRegression
    self.pipe_LogisticRegression = Pipeline([('normalizer', StandardScaler()), 
                                            ('clf', LogisticRegression())])
    self.pipe_LogisticRegression_PCA = Pipeline([('normalizer', StandardScaler()), 
                                                ('PCA', PCA(n_components = self.N_COMPONENTS)), ('clf', LogisticRegression())])
    #RANSACRegressor
    self.pipe_RANSACRegressor = Pipeline([('normalizer', StandardScaler()), 
                                            ('clf', RANSACRegressor())])
    self.pipe_RANSACRegressor_PCA = Pipeline([('normalizer', StandardScaler()), 
                                                ('PCA', PCA(n_components = self.N_COMPONENTS)), ('clf', RANSACRegressor())])
    #Ridge
    self.pipe_Ridge = Pipeline([('normalizer', StandardScaler()), 
                                            ('clf', Ridge())])
    self.pipe_Ridge_PCA = Pipeline([('normalizer', StandardScaler()), 
                                                ('PCA', PCA(n_components = self.N_COMPONENTS)), ('clf', Ridge())])
    #Lasso
    self.pipe_Lasso = Pipeline([('normalizer', StandardScaler()), 
                                            ('clf', Lasso())])
    self.pipe_Ridge_PCA = Pipeline([('normalizer', StandardScaler()), 
                                                ('PCA', PCA(n_components = self.N_COMPONENTS)), ('clf', Lasso())])
    
    self.pipe_KNN_param = [{'clf__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                            'clf__leaf_size': [1, 2, 3, 5],
                            'clf__weights': ['uniform', 'distance'],
                            'algorithm': ['auto', 'ball_tree', 'kdtree', 'brute']}]
    
    self.pipe_SVR_params = [{'clf__kernel': self.KERNELS,
                            'clf__C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                            'clf__gamma': self.GAMMA}]
    
    self.pipe_AdaBoostRegressor_param = [{'clf__n_estimators': np.arange(1,50),
                                          'clf__learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
                                          'clf__loss': ['linear', 'square', 'exponential']}]
    
    
    self.pipe_RandomForestRegressor_params = [{'clf__criterion': self.CRITERION,
                                                 'clf__max_depth': np.arange(2,10),
                                                 'clf__min_samples_split': np.arange(2,10, 1),
                                                 'clf__min_samples_leaf': np.arange(2,10, 1),
                                                 'clf__max_leaf_nodes': np.arange(2, 10, 1)}]

    self.pipe_GradientBoostingRegressor_param = [{'clf__max_depth': np.arange(2,10)}]
    
    self.pipe_GaussianNB_params = [{'clf__priors': [None]}]
    
    self.pipe_GaussianProcessClassifier_params = [{'clf__kernel': [1**2 * RBF(1.0)]}]

    self.pipe_LogisticRegression_params = [{'clf__penalty': ['l1', 'l2'],
                                    		'clf__C': [1.0, 0.5, 0.1], 'clf__solver': ['liblinear']}]

    self.QuadraticDiscriminantAnalysis_params = [{'clf__priors': [None]}]
      
      
      
  def optimize(self, X_train, X_test, Y_train, Y_test):
    '''
    Here we call the GridSearchCV class to get
    the best parameters or better still optimized parameters
    for our data.
    Remember the Gridsearch is done througk the pipeline.
    '''
    
    self.grid_RandomForestClassifier = GridSearchCV(estimator = self.pipe_RandomForestClassifier, 
                                                    param_grid = self.pipe_RandomForestClassifier_params,
                                                    scoring='accuracy',	cv = self.N_VALIDATION)
    
    self.grid_RandomForestClassifier_PCA = GridSearchCV(estimator = self.pipe_RandomForestClassifier_PCA, 
                                                        param_grid = self.pipe_RandomForestClassifier_params,
                                                        scoring='accuracy',	cv = self.N_VALIDATION)
    
    self.grid_KNN = GridSearchCV(estimator = self.pipe_KNN, param_grid = self.pipe_KNN_param,
                                 scoring='accuracy',	cv = self.N_VALIDATION)
    
    self.grid_KNN_PCA = GridSearchCV(estimator = self.pipe_KNN_PCA, param_grid = self.pipe_KNN_param,
                         scoring='accuracy',	cv = self.N_VALIDATION)
    
    self.grid_SVC = GridSearchCV(estimator = self.pipe_SVC, param_grid = self.pipe_SVC_params,
                                         scoring='accuracy',	cv = self.N_VALIDATION)
    
    self.grid_SVC_PCA = GridSearchCV(estimator = self.pipe_SVC_PCA, param_grid = self.pipe_SVC_params,
                                 scoring='accuracy',	cv = self.N_VALIDATION)
    
    self.grid_GaussianProcessClassifier = GridSearchCV(estimator = self.pipe_GaussianProcessClassifier, 
                                                       param_grid = self.pipe_GaussianProcessClassifier_params,
                                                       scoring='accuracy',	cv = self.N_VALIDATION)
    
    self.grid_GaussianProcessClassifier_PCA = GridSearchCV(estimator = self.pipe_GaussianProcessClassifier_PCA, 
                                               param_grid = self.pipe_GaussianProcessClassifier_params,
                                               scoring='accuracy',	cv = self.N_VALIDATION)
    
    self.grid_LogisticRegression = GridSearchCV(estimator = self.pipe_LogisticRegression, 
                                                param_grid = self.pipe_LogisticRegression_params,
                                                scoring='accuracy',	cv = self.N_VALIDATION)
    
    self.grid_LogisticRegression_PCA = GridSearchCV(estimator = self.pipe_LogisticRegression_PCA, 
                                                param_grid = self.pipe_LogisticRegression_params,
                                                scoring='accuracy',	cv = self.N_VALIDATION)
    
    self.grid_DecisionTreeClassifier = GridSearchCV(estimator = self.pipe_DecisionTreeClassifier,
                                                    param_grid = self.pipe_DecisionTreeClassifier_param,
                                                    scoring='accuracy',	cv = self.N_VALIDATION)
    
    self.grid_DecisionTreeClassifier_PCA = GridSearchCV(estimator = self.pipe_DecisionTreeClassifier_PCA,
                                                    param_grid = self.pipe_DecisionTreeClassifier_param,
                                                    scoring='accuracy',	cv = self.N_VALIDATION)
    
    self.grid_AdaBoostClassifier = GridSearchCV(estimator = self.pipe_AdaBoostClassifier,
                                                param_grid = self.pipe_AdaBoostClassifier_param,
                                                scoring='accuracy',	cv = self.N_VALIDATION)
    
    self.grid_AdaBoostClassifier_PCA = GridSearchCV(estimator = self.pipe_AdaBoostClassifier_PCA, 
                                                param_grid = self.pipe_AdaBoostClassifier_param,
                                                scoring='accuracy',	cv = self.N_VALIDATION)
    
    
    self.grid_GaussianNB = GridSearchCV(estimator = self.pipe_GaussianNB, param_grid = self.pipe_GaussianNB_params,
                                        scoring='accuracy',	cv = self.N_VALIDATION)
    
    self.grid_GaussianNB_PCA = GridSearchCV(estimator = self.pipe_GaussianNB_PCA, param_grid = self.pipe_GaussianNB_params,
                                        scoring='accuracy',	cv = self.N_VALIDATION)
    
    self.grid_QuadraticDiscriminantAnalysis = GridSearchCV(estimator = self.pipe_QuadraticDiscriminantAnalysis, 
                                                           param_grid = self.QuadraticDiscriminantAnalysis_params,
                                                           scoring='accuracy',	cv = self.N_VALIDATION)
    
    self.grid_QuadraticDiscriminantAnalysis_PCA = GridSearchCV(estimator = self.pipe_QuadraticDiscriminantAnalysis_PCA, 
                                                           param_grid = self.QuadraticDiscriminantAnalysis_params,
                                                           scoring='accuracy',	cv = self.N_VALIDATION)
    
    
    
    self.All_grids = {'grid_RandomForestClassifier': self.grid_RandomForestClassifier,
                      'grid_RandomForestClassifier_PCA': self.grid_RandomForestClassifier_PCA,
                      'grid_KNN': self.grid_KNN, 'grid_KNN_PCA_': self.grid_KNN_PCA,
                      'grid_SVC': self.grid_SVC, 'grid_SVC_PCA': self.grid_SVC_PCA,
                      'grid_GaussianProcessClassifier': self.grid_GaussianProcessClassifier,
                      'grid_GaussianProcessClassifier_PCA': self.grid_GaussianProcessClassifier_PCA,
                      'grid_LogisticRegression': self.grid_LogisticRegression,
                      'grid_LogisticRegression_PCA': self.grid_LogisticRegression_PCA,
                      'grid_DecisionTreeClassifier': self.grid_DecisionTreeClassifier,
                      'grid_DecisionTreeClassifier_PCA': self.grid_DecisionTreeClassifier_PCA,
                      'grid_AdaBoostClassifier': self.grid_AdaBoostClassifier,
                      'grid_AdaBoostClassifier_PCA': self.grid_AdaBoostClassifier_PCA,
                      'grid_GaussianNB': self.grid_GaussianNB, 
                      'grid_GaussianNB_PCA': self.grid_GaussianNB_PCA,
                      'grid_QuadraticDiscriminantAnalysis': self.grid_QuadraticDiscriminantAnalysis,
                      'grid_QuadraticDiscriminantAnalysis_PCA': self.grid_QuadraticDiscriminantAnalysis_PCA}
    
    print('--------------------------------------------------------')
    print('\tPerforming optimization...')
    
    for classifier_grid_name, classifier_grid in self.All_grids.items():
        print('--------------------------------------------------------')
        print('Classifier: {}'.format(classifier_grid_name))	
    	# Fit grid search	
        classifier_grid.fit(X_train, Y_train)
    	# Best params
        print('Best params: {}'.format(classifier_grid.best_params_))
    	# Best training data accuracy
        print('Best training accuracy: {}'.format(classifier_grid.best_score_))
    	# Predict on test data with best params
        Y_Prediction = classifier_grid.predict(X_test)
    	# Test data accuracy of model with best params
        print('Test set accuracy score for best params: {}'.format(accuracy_score(Y_test, Y_Prediction)))
        print('--------------------------------------------------------')
    	# Track best (highest test accuracy) model
        if accuracy_score(Y_test, Y_Prediction) > self.BEST_ACCURACY:
            self.BEST_ACCURACY = accuracy_score(Y_test, Y_Prediction)
            self.BEST_GRIDSEARCH = classifier_grid
            self.BEST_CLASSIFIER = classifier_grid_name
    print('\nClassifier with best test set accuracy: {}'.format(self.BEST_CLASSIFIER))
    
    return self.BEST_GRIDSEARCH
          