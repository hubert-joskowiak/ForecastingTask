# -*- coding: utf-8 -*-

'''
    File name: machine_learning.py
    Author: Hubert Jóskowiak
    Date created: 27/03/2019
    Python Version: 3.5.5
'''


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import numpy as np
import pprint



def random_forest (train_features, train_labels, test_features):
    
    random_grid = {'n_estimators': [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
                   'max_features': ['auto', 'sqrt'],
                   'max_depth'   : [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                   'min_samples_split': [2, 5, 10],
                   'min_samples_leaf' : [1, 2, 4],
                   'bootstrap'        : [True, False]}
    
    rf = RandomForestRegressor()
    rf_random = RandomizedSearchCV(
            estimator = rf, 
            param_distributions = random_grid, 
            n_iter = 200, 
            cv = 3, 
            scoring = 'neg_mean_squared_error', 
            verbose=5, 
            random_state=42, 
            n_jobs = -1)
    
    rf_result  = rf_random.fit(train_features, train_labels)
    
    print('BEST PARAMS: ')
    pprint.pprint(rf_result.best_params_)
    
    best_model = rf_result.best_estimator_
    predictions = best_model.predict(test_features)
    train_rmse = evaluate(best_model.predict(train_features), train_labels)
    
    return best_model, train_rmse, predictions



def xgboost (train_features, train_labels, test_features):
    
    random_grid = {'n_estimators': [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
                   'max_depth'   : [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                   'eta'         : [0.001, 0.1, 0.3, 0.5],
                   'min_child_weight' : [1, 2, 5, 10]}
    
    xgb = XGBRegressor()
    
    xgb_random = RandomizedSearchCV(
            estimator = xgb, 
            param_distributions = random_grid, 
            n_iter = 200, 
            cv = 3, 
            scoring = 'neg_mean_squared_error', 
            verbose=5, 
            random_state=42, 
            n_jobs = -1)
    
    
    xgb_result = xgb_random.fit(train_features, train_labels)
    
    print('BEST PARAMS: ')
    pprint.pprint(xgb_result.best_params_)
    
    best_model = xgb_result.best_estimator_
    predictions = best_model.predict(test_features)
    train_rmse = evaluate(best_model.predict(train_features), train_labels)
    
    return best_model, train_rmse, predictions



def evaluate(predictions, test_labels):
    
    rmse = np.sqrt(np.mean((predictions-test_labels)**2))
    return rmse


def plot_importances(model, feature_list):
    
    importances = list(model.feature_importances_)
    x_values = list(range(len(importances)))
    plt.bar(x_values, importances, orientation = 'vertical')
    plt.xticks(x_values, feature_list, rotation='vertical')
    plt.bar(x_values, importances, orientation = 'vertical')
    plt.xticks(x_values, feature_list, rotation='vertical')
    plt.ylabel('Importance'); plt.xlabel('Variable')
    
    
