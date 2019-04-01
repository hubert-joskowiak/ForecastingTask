# -*- coding: utf-8 -*-

'''
    File name: data_pre_processing.py
    Author: Hubert Jóskowiak
    Date created: 26/03/2019
    Python Version: 3.5.5
'''

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np




def data_pre_processing(data_path, label_name):

    #LOAD DATA
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data.date)
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['week'] = data['date'].dt.day_name()
    data = data.set_index(['date'])
    data = pd.get_dummies(data)
    
    
    #ANOMALY DETECTION
    print (data.describe())
    anomaly_detection(data)
    
    #check if there are any non-zero total conversion values for zero conversions
    x = data[(data.conversions == 0) & (data.total_conversion_value != 0)]
    data.loc[((data.conversions == 0) & (data.total_conversion_value != 0)), 'total_conversion_value'] = 0
    
    print('Number of nonzero total conversion values for zero conversions: {}'.format(x.shape[0]))
    
    #check if there are any zero price values for non-zero reservations
    x = data[(data.price == 0) & (data.reservations != 0)]
    data.loc[((data.price == 0) & (data.reservations != 0)), 'price'] = 100
    print('Number of zero price values for nonzero reservations: {}'.format(x.shape[0]))

    data = add_baseline_labels(data, label_name)
    
    labels = np.array(data[[label_name, 'baseline']])
    features= data.drop([label_name, 'baseline'], axis = 1)
    feature_list = list(features.columns)
    features = np.array(features)
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.15, random_state = 42)
    baseline_preds = test_labels[:,1]
    test_labels = test_labels[:,0]
    train_labels = train_labels[:,0]

    return train_features, test_features, train_labels, test_labels, baseline_preds, feature_list




def anomaly_detection (data):
    
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(24,16))
    plt.rcParams.update({'font.size': 16})
    
    data['impressions'].plot(ax=axes[0,0], title='impressions')
    data['clicks'].plot(ax=axes[0,1], title = 'clicks')
    data['conversions'].plot(ax=axes[1,0], title = 'conversions')
    data['cost'].plot(ax=axes[1,1], title = 'cost')
    data['total_conversion_value'].plot(ax=axes[2,0], title = 'total_conversion_value')
    data['average_position'].plot(ax=axes[2,1], title = 'average_position')
    data['reservations'].plot(ax=axes[3,0], title = 'reservations')
    data['price'].plot(ax=axes[3,1], title = 'price')
    plt.tight_layout()
    plt.savefig('anomaly_detection.png')
    plt.close(fig)
    print ('\nGraph for each feature as a function of time was saved in file: anomaly_detection.png \n')


def add_baseline_labels(data, label_name):
    
    lookback_range = [1,2,3,4,5,6,7]
    for diff in (lookback_range):
    
        feat_name = 'prev' + str(diff)
        data2 = data.copy()
        x = data2[label_name].shift(periods=diff)
        data[feat_name] = x

    data['baseline'] = data[['prev1', 'prev2', 'prev3', 'prev4', 'prev5', 'prev6', 'prev7']].mean(axis=1).fillna(0)
    data.drop(['prev1', 'prev2', 'prev3', 'prev4', 'prev5', 'prev6', 'prev7'], axis=1, inplace=True)
    
    return data



