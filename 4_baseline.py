import torch
import pickle
import os
import tqdm
import numpy as np
import pandas as pd
import warnings
import recsys_metrics
warnings.filterwarnings('ignore')

from source.baseline import \
    baseline_raw, \
    baseline_ekstrand, \
    baseline_beutel, \
    baseline_zhang, \
    baseline_ziegler

# %% setup parameters
data_names = ['movielens', 'lastfm']
model_names = ['PMF', 'NCF']

baseline_names = ['ziegler']
baselines = [baseline_ziegler]

k = 10
eval_name = {'movielens': ['rmse', 'drmse', 'topic_cover'],
             'lastfm': ['precision', 'recall', 'f_score', 'ndcg', 
                        'dprecision', 'drecall', 'df_score', 'dndcg', 'topic_cover']}

config = {'movielens': {'PMF': {},
                        'NCF': {}},
          'lastfm': {'PMF': {},
                     'NCF': {}}}

# hyperparams
config['movielens']['PMF']['config_model'] = {'factor_num': 16,
                                              'num_layers': 0,
                                              'dropout': 0,
                                              'lambda_param': 0.005}
config['movielens']['PMF']['config_train'] = {'lr': 1e-2,
                                              'epochs': 10000,
                                              'gamma': 0.5,
                                              'batch_size': 32768,
                                              'patience': 1000,
                                              'verbose': True}
config['movielens']['NCF']['config_model'] = {'factor_num': 16,
                                              'num_layers': 2,
                                              'dropout': 0.2,
                                              'lambda_param': 0.0025}
config['movielens']['NCF']['config_train'] = {'lr': 1e-2,
                                              'epochs': 10000,
                                              'gamma': 0.5,
                                              'batch_size': 32768,
                                              'patience': 1000,
                                              'verbose': True}
config['lastfm']['PMF']['config_model'] = {'factor_num': 32,
                                           'num_layers': 0,
                                           'dropout': 0,
                                           'lambda_param': 0}
config['lastfm']['PMF']['config_train'] = {'lr': 1e-2,
                                           'epochs': 10000,
                                           'gamma': 0.5,
                                           'batch_size': 32768,
                                           'patience': 1000,
                                           'verbose': True}
config['lastfm']['NCF']['config_model'] = {'factor_num': 16,
                                           'num_layers': 2,
                                           'dropout': 0.2,
                                           'lambda_param': 0}
config['lastfm']['NCF']['config_train'] = {'lr': 1e-2,
                                           'epochs': 10000,
                                           'gamma': 0.5,
                                           'batch_size': 32768,
                                           'patience': 1000,
                                           'verbose': True}

# %% run the experiment
for i, baseline_class in enumerate(baselines):
    baseline_performance = {'movielens': {'PMF': {},
                                          'NCF': {}},
                            'lastfm': {'PMF': {},
                                       'NCF': {}}}
    for data_name in data_names:
        for model_name in model_names:
        
            
            # baseline name
            baseline_name = baseline_names[i]
            
            # run
            torch.manual_seed(2023)
            baseline = baseline_class(data_name = data_name, model_name = model_name, k = k,
                                      config_model = config[data_name][model_name]['config_model'],
                                      config_train = config[data_name][model_name]['config_train'])
            
            # save
            baseline_performance[data_name][model_name] = baseline.exp.eval_test
            
    with open('4_baseline/output/'+str(baseline_name), 'wb') as f:
        pickle.dump(baseline_performance, f)
            
