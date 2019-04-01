# -*- coding: utf-8 -*-

'''
    File name: main.py
    Author: Hubert Jóskowiak
    Date created: 27/03/2019
    Python Version: 3.5.5
'''


from data_pre_processing import data_pre_processing
from machine_learning import random_forest, xgboost, evaluate, plot_importances
from neural_network import dnn
import matplotlib.pyplot as plt


data_path  = 'ad_data_daily.csv'
label_name = 'conversions'

train_features, test_features, train_labels, test_labels, baseline_preds, feature_list = data_pre_processing(data_path, label_name)

### BASELINE ###
base_rmse = evaluate(baseline_preds, test_labels)


### RANDOM FOREST ###
rf_model, rf_rmse_train, rf_predictions = random_forest(train_features, train_labels, test_features)
rf_rmse = evaluate(rf_predictions, test_labels)


### XGBOOST ###
xgb_model, xgb_rmse_train, xgb_predictions = xgboost(train_features, train_labels, test_features)
xgb_rmse = evaluate(xgb_predictions, test_labels)


### NEURAL NETWORK ###
nn_predictions, cost = dnn(train_features, train_labels, test_features)
nn_rmse = evaluate(nn_predictions, test_labels)


print('BASELINE rmse       : ', base_rmse)
print('RANDOM FOREST rmse  : ', rf_rmse)
print('XGBOOST rmse        : ', xgb_rmse)
print('NEURAL NETWORK rmse : ', nn_rmse)



print('\n Variable Importances for Random Forest Classifier')
plot_importances(rf_model, feature_list)
plt.show()
print('\n Variable Importances for XGBoost Classifier')
plot_importances(xgb_model, feature_list)
plt.show()
print('\n RMSE Value in DNN Training')
plt.figure(figsize=(12,4))
plt.ylabel('RMSE'); plt.xlabel('Epoch')
plt.plot(cost)
plt.show()


