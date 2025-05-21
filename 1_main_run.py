import torch
import pickle
import os
import tqdm
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

os.chdir('/home/u9/tianyangxie/Documents/cf')
from source.experiment import experiment

# %% setup parameters
print('Data name? (e.g. movielens, lastfm)')
data_names = input().replace(' ','').split(',')
print('Model name? (e.g. PMF, NCF)')
model_names = input().replace(' ','').split(',')
print('Loss name? (e.g. A, A+F+D)')
loss_names = input().replace(' ','').split(',')

k = 10
eval_name = {'movielens': ['rmse', 'drmse', 'topic_cover'],
             'lastfm': ['precision', 'recall', 'f_score', 'ndcg', 
                        'dprecision', 'drecall', 'df_score', 'dndcg', 'topic_cover']}
proxy_name = ['A', 'F', 'D']

config = {'movielens': {'PMF': {'A': {}, 'A+F': {}, 'A+D': {}, 'A+F+D':{}},
                        'NCF': {'A': {}, 'A+F': {}, 'A+D': {}, 'A+F+D':{}}},
          'lastfm': {'PMF': {'A': {}, 'A+F': {}, 'A+D': {}, 'A+F+D':{}},
                     'NCF': {'A': {}, 'A+F': {}, 'A+D': {}, 'A+F+D':{}}}}

# movielens_PMF_A
config['movielens']['PMF']['A']['config_model'] = {'factor_num': 16,
                                                   'num_layers': 0,
                                                   'dropout': 0,
                                                   'lambda_param': 0.005, 
                                                   'lambda_F': 0,
                                                   'lambda_D': 0,
                                                   'contrast': 'bpr'}
config['movielens']['PMF']['A']['config_augment'] = {'r': 0.01,
                                                     't': 100,
                                                     'thres': [0, 0, 0]}
config['movielens']['PMF']['A']['config_initial_train'] = {'lr': 1e-2,
                                                           'epochs': 10000,
                                                           'gamma': 0.5,
                                                           'batch_size': 32768,
                                                           'patience': 1000,
                                                           'verbose': True}
config['movielens']['PMF']['A']['config_dynamic_train'] = {'lr': 1e-4,
                                                           'epochs': 1000,
                                                           'gamma': 0.5,
                                                           'batch_size': 32768,
                                                           'patience': 200,
                                                           'verbose': True}
config['movielens']['PMF']['A']['config_initial_influence'] = {'batch_size_first': 32768,
                                                               'batch_size_second': 32768,
                                                               'batch_size_third': 1248,
                                                               'epochs_first': 1000,
                                                               'epochs_second': 1000,
                                                               'epochs_check': 100,
                                                               'lr_second': 1,
                                                               'warmup_second': False}
config['movielens']['PMF']['A']['config_dynamic_influence'] = {'batch_size_first': 32768,
                                                               'batch_size_second': 32768,
                                                               'batch_size_third': 1248,
                                                               'epochs_first': 1000,
                                                               'epochs_second': 100,
                                                               'epochs_check': 100,
                                                               'lr_second': 1,
                                                               'warmup_second': True}

# movielens_PMF_A+F
config['movielens']['PMF']['A+F']['config_model'] = {'factor_num': 16,
                                                   'num_layers': 0,
                                                   'dropout': 0,
                                                   'lambda_param': 0.005, 
                                                   'lambda_F': 0.001,
                                                   'lambda_D': 0,
                                                   'contrast': 'bpr'}
config['movielens']['PMF']['A+F']['config_augment'] = {'r': 0.01,
                                                     't': 100,
                                                     'thres': [0, 0, 0]}
config['movielens']['PMF']['A+F']['config_initial_train'] = {'lr': 1e-2,
                                                           'epochs': 10000,
                                                           'gamma': 0.5,
                                                           'batch_size': 32768,
                                                           'patience': 1000,
                                                           'verbose': True}
config['movielens']['PMF']['A+F']['config_dynamic_train'] = {'lr': 1e-4,
                                                           'epochs': 1000,
                                                           'gamma': 0.5,
                                                           'batch_size': 32768,
                                                           'patience': 200,
                                                           'verbose': True}
config['movielens']['PMF']['A+F']['config_initial_influence'] = {'batch_size_first': 32768,
                                                               'batch_size_second': 32768,
                                                               'batch_size_third': 1248,
                                                               'epochs_first': 1000,
                                                               'epochs_second': 1000,
                                                               'epochs_check': 100,
                                                               'lr_second': 1,
                                                               'warmup_second': False}
config['movielens']['PMF']['A+F']['config_dynamic_influence'] = {'batch_size_first': 32768,
                                                               'batch_size_second': 32768,
                                                               'batch_size_third': 1248,
                                                               'epochs_first': 1000,
                                                               'epochs_second': 100,
                                                               'epochs_check': 100,
                                                               'lr_second': 1,
                                                               'warmup_second': True}

# movielens_PMF_A+D
config['movielens']['PMF']['A+D']['config_model'] = {'factor_num': 16,
                                                   'num_layers': 0,
                                                   'dropout': 0,
                                                   'lambda_param': 0.005, 
                                                   'lambda_F': 0,
                                                   'lambda_D': 0.0001,
                                                   'contrast': 'bpr'}
config['movielens']['PMF']['A+D']['config_augment'] = {'r': 0.01,
                                                     't': 100,
                                                     'thres': [0, 0, 0]}
config['movielens']['PMF']['A+D']['config_initial_train'] = {'lr': 1e-2,
                                                           'epochs': 10000,
                                                           'gamma': 0.5,
                                                           'batch_size': 32768,
                                                           'patience': 1000,
                                                           'verbose': True}
config['movielens']['PMF']['A+D']['config_dynamic_train'] = {'lr': 1e-4,
                                                           'epochs': 1000,
                                                           'gamma': 0.5,
                                                           'batch_size': 32768,
                                                           'patience': 200,
                                                           'verbose': True}
config['movielens']['PMF']['A+D']['config_initial_influence'] = {'batch_size_first': 32768,
                                                               'batch_size_second': 32768,
                                                               'batch_size_third': 1248,
                                                               'epochs_first': 1000,
                                                               'epochs_second': 1000,
                                                               'epochs_check': 100,
                                                               'lr_second': 1,
                                                               'warmup_second': False}
config['movielens']['PMF']['A+D']['config_dynamic_influence'] = {'batch_size_first': 32768,
                                                               'batch_size_second': 32768,
                                                               'batch_size_third': 1248,
                                                               'epochs_first': 1000,
                                                               'epochs_second': 100,
                                                               'epochs_check': 100,
                                                               'lr_second': 1,
                                                               'warmup_second': True}

# movielens_PMF_A+F+D
config['movielens']['PMF']['A+F+D']['config_model'] = {'factor_num': 16,
                                                       'num_layers': 0,
                                                       'dropout': 0,
                                                       'lambda_param': 0.005, 
                                                       'lambda_F': 0.0001,
                                                       'lambda_D': 0.0001,
                                                       'contrast': 'bpr'}
config['movielens']['PMF']['A+F+D']['config_augment'] = {'r': 0.01,
                                                         't': 100,
                                                         'thres': [0, 0, 0]}
config['movielens']['PMF']['A+F+D']['config_initial_train'] = {'lr': 1e-2,
                                                               'epochs': 10000,
                                                               'gamma': 0.5,
                                                               'batch_size': 32768,
                                                               'patience': 1000,
                                                               'verbose': True}
config['movielens']['PMF']['A+F+D']['config_dynamic_train'] = {'lr': 1e-4,
                                                               'epochs': 1000,
                                                               'gamma': 0.5,
                                                               'batch_size': 32768,
                                                               'patience': 200,
                                                               'verbose': True}
config['movielens']['PMF']['A+F+D']['config_initial_influence'] = {'batch_size_first': 32768,
                                                                   'batch_size_second': 32768,
                                                                   'batch_size_third': 1248,
                                                                   'epochs_first': 1000,
                                                                   'epochs_second': 1000,
                                                                   'epochs_check': 100,
                                                                   'lr_second': 1,
                                                                   'warmup_second': False}
config['movielens']['PMF']['A+F+D']['config_dynamic_influence'] = {'batch_size_first': 32768,
                                                                   'batch_size_second': 32768,
                                                                   'batch_size_third': 1248,
                                                                   'epochs_first': 1000,
                                                                   'epochs_second': 100,
                                                                   'epochs_check': 100,
                                                                   'lr_second': 1,
                                                                   'warmup_second': True}

# movielens_NCF_A
config['movielens']['NCF']['A']['config_model'] = {'factor_num': 16,
                                                   'num_layers': 2,
                                                   'dropout': 0.2,
                                                   'lambda_param': 0.0025, 
                                                   'lambda_F': 0,
                                                   'lambda_D': 0,
                                                   'contrast': 'bpr'}
config['movielens']['NCF']['A']['config_augment'] = {'r': 0.005,
                                                     't': 100,
                                                     'thres': [0, 0, 0]}
config['movielens']['NCF']['A']['config_initial_train'] = {'lr': 1e-2,
                                                           'epochs': 10000,
                                                           'gamma': 0.5,
                                                           'batch_size': 32768,
                                                           'patience': 1000,
                                                           'verbose': True}
config['movielens']['NCF']['A']['config_dynamic_train'] = {'lr': 1e-4,
                                                           'epochs': 1000,
                                                           'gamma': 0.5,
                                                           'batch_size': 32768,
                                                           'patience': 200,
                                                           'verbose': True}
config['movielens']['NCF']['A']['config_initial_influence'] = {'batch_size_first': 32768,
                                                               'batch_size_second': 32768,
                                                               'batch_size_third': 1248,
                                                               'epochs_first': 1000,
                                                               'epochs_second': 10000,
                                                               'epochs_check': 100,
                                                               'lr_second': 0.01,
                                                               'warmup_second': False}
config['movielens']['NCF']['A']['config_dynamic_influence'] = {'batch_size_first': 32768,
                                                               'batch_size_second': 32768,
                                                               'batch_size_third': 1248,
                                                               'epochs_first': 1000,
                                                               'epochs_second': 10000,
                                                               'epochs_check': 100,
                                                               'lr_second': 0.01,
                                                               'warmup_second': False}

# movielens_NCF_A+F
config['movielens']['NCF']['A+F']['config_model'] = {'factor_num': 16,
                                                   'num_layers': 2,
                                                   'dropout': 0.2,
                                                   'lambda_param': 0.0025, 
                                                   'lambda_F': 0.001,
                                                   'lambda_D': 0,
                                                   'contrast': 'bpr'}
config['movielens']['NCF']['A+F']['config_augment'] = {'r': 0.005,
                                                     't': 100,
                                                     'thres': [0, 0, 0]}
config['movielens']['NCF']['A+F']['config_initial_train'] = {'lr': 1e-2,
                                                           'epochs': 10000,
                                                           'gamma': 0.5,
                                                           'batch_size': 32768,
                                                           'patience': 1000,
                                                           'verbose': True}
config['movielens']['NCF']['A+F']['config_dynamic_train'] = {'lr': 1e-4,
                                                           'epochs': 1000,
                                                           'gamma': 0.5,
                                                           'batch_size': 32768,
                                                           'patience': 200,
                                                           'verbose': True}
config['movielens']['NCF']['A+F']['config_initial_influence'] = {'batch_size_first': 32768,
                                                               'batch_size_second': 32768,
                                                               'batch_size_third': 1248,
                                                               'epochs_first': 1000,
                                                               'epochs_second': 10000,
                                                               'epochs_check': 100,
                                                               'lr_second': 0.01,
                                                               'warmup_second': False}
config['movielens']['NCF']['A+F']['config_dynamic_influence'] = {'batch_size_first': 32768,
                                                               'batch_size_second': 32768,
                                                               'batch_size_third': 1248,
                                                               'epochs_first': 1000,
                                                               'epochs_second': 10000,
                                                               'epochs_check': 100,
                                                               'lr_second': 0.01,
                                                               'warmup_second': False}

# movielens_NCF_A+D
config['movielens']['NCF']['A+D']['config_model'] = {'factor_num': 16,
                                                   'num_layers': 2,
                                                   'dropout': 0.2,
                                                   'lambda_param': 0.0025, 
                                                   'lambda_F': 0,
                                                   'lambda_D': 0.01,
                                                   'contrast': 'bpr'}
config['movielens']['NCF']['A+D']['config_augment'] = {'r': 0.005,
                                                     't': 100,
                                                     'thres': [0, 0, 0]}
config['movielens']['NCF']['A+D']['config_initial_train'] = {'lr': 1e-2,
                                                           'epochs': 10000,
                                                           'gamma': 0.5,
                                                           'batch_size': 32768,
                                                           'patience': 1000,
                                                           'verbose': True}
config['movielens']['NCF']['A+D']['config_dynamic_train'] = {'lr': 1e-4,
                                                           'epochs': 1000,
                                                           'gamma': 0.5,
                                                           'batch_size': 32768,
                                                           'patience': 200,
                                                           'verbose': True}
config['movielens']['NCF']['A+D']['config_initial_influence'] = {'batch_size_first': 32768,
                                                               'batch_size_second': 32768,
                                                               'batch_size_third': 1248,
                                                               'epochs_first': 1000,
                                                               'epochs_second': 10000,
                                                               'epochs_check': 100,
                                                               'lr_second': 0.01,
                                                               'warmup_second': False}
config['movielens']['NCF']['A+D']['config_dynamic_influence'] = {'batch_size_first': 32768,
                                                               'batch_size_second': 32768,
                                                               'batch_size_third': 1248,
                                                               'epochs_first': 1000,
                                                               'epochs_second': 10000,
                                                               'epochs_check': 100,
                                                               'lr_second': 0.01,
                                                               'warmup_second': False}
    
# movielens_NCF_A+F+D
config['movielens']['NCF']['A+F+D']['config_model'] = {'factor_num': 16,
                                                       'num_layers': 2,
                                                       'dropout': 0.2,
                                                       'lambda_param': 0.0025, 
                                                       'lambda_F': 0.0001,
                                                       'lambda_D': 0.01,
                                                       'contrast': 'bpr'}
config['movielens']['NCF']['A+F+D']['config_augment'] = {'r': 0.005,
                                                         't': 100,
                                                         'thres': [0, 0, 0]}
config['movielens']['NCF']['A+F+D']['config_initial_train'] = {'lr': 1e-2,
                                                               'epochs': 10000,
                                                               'gamma': 0.5,
                                                               'batch_size': 32768,
                                                               'patience': 1000,
                                                               'verbose': True}
config['movielens']['NCF']['A+F+D']['config_dynamic_train'] = {'lr': 1e-4,
                                                               'epochs': 1000,
                                                               'gamma': 0.5,
                                                               'batch_size': 32768,
                                                               'patience': 200,
                                                               'verbose': True}
config['movielens']['NCF']['A+F+D']['config_initial_influence'] = {'batch_size_first': 32768,
                                                                   'batch_size_second': 32768,
                                                                   'batch_size_third': 1248,
                                                                   'epochs_first': 1000,
                                                                   'epochs_second': 10000,
                                                                   'epochs_check': 100,
                                                                   'lr_second': 0.01,
                                                                   'warmup_second': False}
config['movielens']['NCF']['A+F+D']['config_dynamic_influence'] = {'batch_size_first': 32768,
                                                                   'batch_size_second': 32768,
                                                                   'batch_size_third': 1248,
                                                                   'epochs_first': 1000,
                                                                   'epochs_second': 10000,
                                                                   'epochs_check': 100,
                                                                   'lr_second': 0.01,
                                                                   'warmup_second': False}

# lastfm_PMF_A
config['lastfm']['PMF']['A']['config_model'] = {'factor_num': 32,
                                                'num_layers': 0,
                                                'dropout': 0,
                                                'lambda_param': 0, 
                                                'lambda_F': 0,
                                                'lambda_D': 0,
                                                'contrast': 'bpr'}
config['lastfm']['PMF']['A']['config_augment'] = {'r': 0.005,
                                                  't': 1000,
                                                  'thres': [1, -1, 1]}
config['lastfm']['PMF']['A']['config_initial_train'] = {'lr': 1e-2,
                                                        'epochs': 10000,
                                                        'gamma': 0.5,
                                                        'batch_size': 32768,
                                                        'patience': 1000,
                                                        'verbose': True}
config['lastfm']['PMF']['A']['config_dynamic_train'] = {'lr': 1e-4,
                                                        'epochs': 1000,
                                                        'gamma': 0.5,
                                                        'batch_size': 32768,
                                                        'patience': 200,
                                                        'verbose': True}
config['lastfm']['PMF']['A']['config_initial_influence'] = {'batch_size_first': 32768,
                                                            'batch_size_second': 32768,
                                                            'batch_size_third': 1248,
                                                            'epochs_first': 1000,
                                                            'epochs_second': 1000,
                                                            'epochs_check': 100,
                                                            'lr_second': 1,
                                                            'warmup_second': False}
config['lastfm']['PMF']['A']['config_dynamic_influence'] = {'batch_size_first': 32768,
                                                            'batch_size_second': 32768,
                                                            'batch_size_third': 1248,
                                                            'epochs_first': 1000,
                                                            'epochs_second': 100,
                                                            'epochs_check': 100,
                                                            'lr_second': 1,
                                                            'warmup_second': True}

# lastfm_PMF_A+F
config['lastfm']['PMF']['A+F']['config_model'] = {'factor_num': 32,
                                                'num_layers': 0,
                                                'dropout': 0,
                                                'lambda_param': 0, 
                                                'lambda_F': 0.001,
                                                'lambda_D': 0,
                                                'contrast': 'bpr'}
config['lastfm']['PMF']['A+F']['config_augment'] = {'r': 0.005,
                                                  't': 1000,
                                                  'thres': [1, -1, 1]}
config['lastfm']['PMF']['A+F']['config_initial_train'] = {'lr': 1e-2,
                                                        'epochs': 10000,
                                                        'gamma': 0.5,
                                                        'batch_size': 32768,
                                                        'patience': 1000,
                                                        'verbose': True}
config['lastfm']['PMF']['A+F']['config_dynamic_train'] = {'lr': 1e-4,
                                                        'epochs': 1000,
                                                        'gamma': 0.5,
                                                        'batch_size': 32768,
                                                        'patience': 200,
                                                        'verbose': True}
config['lastfm']['PMF']['A+F']['config_initial_influence'] = {'batch_size_first': 32768,
                                                            'batch_size_second': 32768,
                                                            'batch_size_third': 1248,
                                                            'epochs_first': 1000,
                                                            'epochs_second': 1000,
                                                            'epochs_check': 100,
                                                            'lr_second': 1,
                                                            'warmup_second': False}
config['lastfm']['PMF']['A+F']['config_dynamic_influence'] = {'batch_size_first': 32768,
                                                            'batch_size_second': 32768,
                                                            'batch_size_third': 1248,
                                                            'epochs_first': 1000,
                                                            'epochs_second': 100,
                                                            'epochs_check': 100,
                                                            'lr_second': 1,
                                                            'warmup_second': True}

# lastfm_PMF_A+D
config['lastfm']['PMF']['A+D']['config_model'] = {'factor_num': 32,
                                                'num_layers': 0,
                                                'dropout': 0,
                                                'lambda_param': 0, 
                                                'lambda_F': 0,
                                                'lambda_D': 0.001,
                                                'contrast': 'bpr'}
config['lastfm']['PMF']['A+D']['config_augment'] = {'r': 0.005,
                                                  't': 1000,
                                                  'thres': [1, -1, 1]}
config['lastfm']['PMF']['A+D']['config_initial_train'] = {'lr': 1e-2,
                                                        'epochs': 10000,
                                                        'gamma': 0.5,
                                                        'batch_size': 32768,
                                                        'patience': 1000,
                                                        'verbose': True}
config['lastfm']['PMF']['A+D']['config_dynamic_train'] = {'lr': 1e-4,
                                                        'epochs': 1000,
                                                        'gamma': 0.5,
                                                        'batch_size': 32768,
                                                        'patience': 200,
                                                        'verbose': True}
config['lastfm']['PMF']['A+D']['config_initial_influence'] = {'batch_size_first': 32768,
                                                            'batch_size_second': 32768,
                                                            'batch_size_third': 1248,
                                                            'epochs_first': 1000,
                                                            'epochs_second': 1000,
                                                            'epochs_check': 100,
                                                            'lr_second': 1,
                                                            'warmup_second': False}
config['lastfm']['PMF']['A+D']['config_dynamic_influence'] = {'batch_size_first': 32768,
                                                            'batch_size_second': 32768,
                                                            'batch_size_third': 1248,
                                                            'epochs_first': 1000,
                                                            'epochs_second': 100,
                                                            'epochs_check': 100,
                                                            'lr_second': 1,
                                                            'warmup_second': True}

# lastfm_PMF_A+F+D
config['lastfm']['PMF']['A+F+D']['config_model'] = {'factor_num': 32,
                                                    'num_layers': 0,
                                                    'dropout': 0,
                                                    'lambda_param': 0, 
                                                    'lambda_F': 0.0001,
                                                    'lambda_D': 0.1,
                                                    'contrast': 'bpr'}
config['lastfm']['PMF']['A+F+D']['config_augment'] = {'r': 0.005,
                                                      't': 1000,
                                                      'thres': [1, -1, 1]}
config['lastfm']['PMF']['A+F+D']['config_initial_train'] = {'lr': 1e-2,
                                                            'epochs': 10000,
                                                            'gamma': 0.5,
                                                            'batch_size': 32768,
                                                            'patience': 1000,
                                                            'verbose': True}
config['lastfm']['PMF']['A+F+D']['config_dynamic_train'] = {'lr': 1e-4,
                                                            'epochs': 1000,
                                                            'gamma': 0.5,
                                                            'batch_size': 32768,
                                                            'patience': 200,
                                                            'verbose': True}
config['lastfm']['PMF']['A+F+D']['config_initial_influence'] = {'batch_size_first': 32768,
                                                                'batch_size_second': 32768,
                                                                'batch_size_third': 1248,
                                                                'epochs_first': 1000,
                                                                'epochs_second': 1000,
                                                                'epochs_check': 100,
                                                                'lr_second': 1,
                                                                'warmup_second': False}
config['lastfm']['PMF']['A+F+D']['config_dynamic_influence'] = {'batch_size_first': 32768,
                                                                'batch_size_second': 32768,
                                                                'batch_size_third': 1248,
                                                                'epochs_first': 1000,
                                                                'epochs_second': 100,
                                                                'epochs_check': 100,
                                                                'lr_second': 1,
                                                                'warmup_second': True}

# lastfm_NCF_A
config['lastfm']['NCF']['A']['config_model'] = {'factor_num': 16,
                                                'num_layers': 2,
                                                'dropout': 0.2,
                                                'lambda_param': 0, 
                                                'lambda_F': 0,
                                                'lambda_D': 0,
                                                'contrast': 'bpr'}
config['lastfm']['NCF']['A']['config_augment'] = {'r': 0.05,
                                                  't': 100,
                                                  'thres': [1, -1, 1]}
config['lastfm']['NCF']['A']['config_initial_train'] = {'lr': 1e-2,
                                                        'epochs': 10000,
                                                        'gamma': 0.5,
                                                        'batch_size': 32768,
                                                        'patience': 1000,
                                                        'verbose': True}
config['lastfm']['NCF']['A']['config_dynamic_train'] = {'lr': 1e-4,
                                                        'epochs': 1000,
                                                        'gamma': 0.5,
                                                        'batch_size': 32768,
                                                        'patience': 200,
                                                        'verbose': True}
config['lastfm']['NCF']['A']['config_initial_influence'] = {'batch_size_first': 32768,
                                                            'batch_size_second': 32768,
                                                            'batch_size_third': 1248,
                                                            'epochs_first': 1000,
                                                            'epochs_second': 1000,
                                                            'epochs_check': 100,
                                                            'lr_second': 1,
                                                            'warmup_second': False}
config['lastfm']['NCF']['A']['config_dynamic_influence'] = {'batch_size_first': 32768,
                                                            'batch_size_second': 32768,
                                                            'batch_size_third': 1248,
                                                            'epochs_first': 1000,
                                                            'epochs_second': 1000,
                                                            'epochs_check': 100,
                                                            'lr_second': 1,
                                                            'warmup_second': False}

# lastfm_NCF_A+F
config['lastfm']['NCF']['A+F']['config_model'] = {'factor_num': 16,
                                                'num_layers': 2,
                                                'dropout': 0.2,
                                                'lambda_param': 0, 
                                                'lambda_F': 0.001,
                                                'lambda_D': 0,
                                                'contrast': 'bpr'}
config['lastfm']['NCF']['A+F']['config_augment'] = {'r': 0.05,
                                                  't': 100,
                                                  'thres': [1, -1, 1]}
config['lastfm']['NCF']['A+F']['config_initial_train'] = {'lr': 1e-2,
                                                        'epochs': 10000,
                                                        'gamma': 0.5,
                                                        'batch_size': 32768,
                                                        'patience': 1000,
                                                        'verbose': True}
config['lastfm']['NCF']['A+F']['config_dynamic_train'] = {'lr': 1e-4,
                                                        'epochs': 1000,
                                                        'gamma': 0.5,
                                                        'batch_size': 32768,
                                                        'patience': 200,
                                                        'verbose': True}
config['lastfm']['NCF']['A+F']['config_initial_influence'] = {'batch_size_first': 32768,
                                                            'batch_size_second': 32768,
                                                            'batch_size_third': 1248,
                                                            'epochs_first': 1000,
                                                            'epochs_second': 1000,
                                                            'epochs_check': 100,
                                                            'lr_second': 1,
                                                            'warmup_second': False}
config['lastfm']['NCF']['A+F']['config_dynamic_influence'] = {'batch_size_first': 32768,
                                                            'batch_size_second': 32768,
                                                            'batch_size_third': 1248,
                                                            'epochs_first': 1000,
                                                            'epochs_second': 1000,
                                                            'epochs_check': 100,
                                                            'lr_second': 1,
                                                            'warmup_second': False}

# lastfm_NCF_A+D
config['lastfm']['NCF']['A+D']['config_model'] = {'factor_num': 16,
                                                'num_layers': 2,
                                                'dropout': 0.2,
                                                'lambda_param': 0, 
                                                'lambda_F': 0,
                                                'lambda_D': 0.1,
                                                'contrast': 'bpr'}
config['lastfm']['NCF']['A+D']['config_augment'] = {'r': 0.05,
                                                  't': 100,
                                                  'thres': [1, -1, 1]}
config['lastfm']['NCF']['A+D']['config_initial_train'] = {'lr': 1e-2,
                                                        'epochs': 10000,
                                                        'gamma': 0.5,
                                                        'batch_size': 32768,
                                                        'patience': 1000,
                                                        'verbose': True}
config['lastfm']['NCF']['A+D']['config_dynamic_train'] = {'lr': 1e-4,
                                                        'epochs': 1000,
                                                        'gamma': 0.5,
                                                        'batch_size': 32768,
                                                        'patience': 200,
                                                        'verbose': True}
config['lastfm']['NCF']['A+D']['config_initial_influence'] = {'batch_size_first': 32768,
                                                            'batch_size_second': 32768,
                                                            'batch_size_third': 1248,
                                                            'epochs_first': 1000,
                                                            'epochs_second': 1000,
                                                            'epochs_check': 100,
                                                            'lr_second': 1,
                                                            'warmup_second': False}
config['lastfm']['NCF']['A+D']['config_dynamic_influence'] = {'batch_size_first': 32768,
                                                            'batch_size_second': 32768,
                                                            'batch_size_third': 1248,
                                                            'epochs_first': 1000,
                                                            'epochs_second': 1000,
                                                            'epochs_check': 100,
                                                            'lr_second': 1,
                                                            'warmup_second': False}
    
# lastfm_NCF_A+F+D
config['lastfm']['NCF']['A+F+D']['config_model'] = {'factor_num': 16,
                                                    'num_layers': 2,
                                                    'dropout': 0.2,
                                                    'lambda_param': 0, 
                                                    'lambda_F': 0.01,
                                                    'lambda_D': 0.1,
                                                    'contrast': 'bpr'}
config['lastfm']['NCF']['A+F+D']['config_augment'] = {'r': 0.05,
                                                      't': 100,
                                                      'thres': [1, -1, 1]}
config['lastfm']['NCF']['A+F+D']['config_initial_train'] = {'lr': 1e-2,
                                                            'epochs': 10000,
                                                            'gamma': 0.5,
                                                            'batch_size': 32768,
                                                            'patience': 1000,
                                                            'verbose': True}
config['lastfm']['NCF']['A+F+D']['config_dynamic_train'] = {'lr': 1e-4,
                                                            'epochs': 1000,
                                                            'gamma': 0.5,
                                                            'batch_size': 32768,
                                                            'patience': 200,
                                                            'verbose': True}
config['lastfm']['NCF']['A+F+D']['config_initial_influence'] = {'batch_size_first': 32768,
                                                                'batch_size_second': 32768,
                                                                'batch_size_third': 1248,
                                                                'epochs_first': 1000,
                                                                'epochs_second': 1000,
                                                                'epochs_check': 100,
                                                                'lr_second': 1,
                                                                'warmup_second': False}
config['lastfm']['NCF']['A+F+D']['config_dynamic_influence'] = {'batch_size_first': 32768,
                                                                'batch_size_second': 32768,
                                                                'batch_size_third': 1248,
                                                                'epochs_first': 1000,
                                                                'epochs_second': 1000,
                                                                'epochs_check': 100,
                                                                'lr_second': 1,
                                                                'warmup_second': False}

# %% run the experiment
for data_name in data_names:
    for model_name in model_names:
        for loss_name in loss_names:
            
            # prepare
            factor_num = config[data_name][model_name][loss_name]['config_model']['factor_num']
            num_layers = config[data_name][model_name][loss_name]['config_model']['num_layers']
            dropout = config[data_name][model_name][loss_name]['config_model']['dropout']
            lambda_param = config[data_name][model_name][loss_name]['config_model']['lambda_param']
            lambda_F = config[data_name][model_name][loss_name]['config_model']['lambda_F']
            lambda_D = config[data_name][model_name][loss_name]['config_model']['lambda_D']
            contrast = config[data_name][model_name][loss_name]['config_model']['contrast']
            r = config[data_name][model_name][loss_name]['config_augment']['r']
            t = config[data_name][model_name][loss_name]['config_augment']['t']
            thres = config[data_name][model_name][loss_name]['config_augment']['thres']
            config_initial_train = config[data_name][model_name][loss_name]['config_initial_train']
            config_dynamic_train = config[data_name][model_name][loss_name]['config_dynamic_train']
            config_initial_influence = config[data_name][model_name][loss_name]['config_initial_influence']
            config_dynamic_influence = config[data_name][model_name][loss_name]['config_dynamic_influence']
            
            eval_test_list, eval_val_list, proxy_val_list = [], [], []
            
            # run
            torch.manual_seed(2023)
            exp = experiment(data_name = data_name, model_name = model_name, loss_name = loss_name,
                             device = 'cuda', contrast = contrast,
                             factor_num = factor_num, num_layers = num_layers, dropout = dropout,
                             r = r, beta = 1,
                             lambda_param = lambda_param, lambda_F = lambda_F, lambda_D = lambda_D)
            exp.prepare_model()
            exp.fit(config = config_initial_train)
            exp.evaluate(k=k)
            exp.monitor()
            eval_test_list.append(exp.eval_test)
            eval_val_list.append(exp.eval_val)
            proxy_val_list.append(exp.proxy_val)
            for i in range(t):
                exp.prepare_candidates()
                if i == 0:
                    exp.prepare_influence(config = config_initial_influence)
                else:
                    exp.prepare_influence(config = config_dynamic_influence)
                exp.prepare_selection(thres)
                exp.fit(config = config_dynamic_train)
                exp.evaluate(k=k)
                exp.monitor()
                eval_test_list.append(exp.eval_test)
                eval_val_list.append(exp.eval_val)
                proxy_val_list.append(exp.proxy_val)
                
            # save
            eval_test, eval_val, proxy_val = \
                np.array(eval_test_list), \
                np.array(eval_val_list), \
                np.array([item.tolist() for item in proxy_val_list])
            eval_test_p, eval_val_p, proxy_val_p = \
                eval_test/eval_test[0, :], \
                eval_val/eval_val[0, :], \
                proxy_val/proxy_val[0, :]
            pd.DataFrame(eval_test, columns = eval_name[data_name])\
                .to_csv("1_main/output/"+ data_name + '_' + model_name + '_' + loss_name +\
                        '_eval_test_real.csv', index = False)
            pd.DataFrame(eval_val, columns = eval_name[data_name])\
                .to_csv("1_main/output/"+ data_name + '_' + model_name + '_' + loss_name +\
                        '_eval_val_real.csv', index = False)
            pd.DataFrame(proxy_val, columns = proxy_name)\
                .to_csv("1_main/output/"+ data_name + '_' + model_name + '_' + loss_name +\
                        '_proxy_val_real.csv', index = False)
            pd.DataFrame(eval_test_p, columns = eval_name[data_name])\
                .to_csv("1_main/output/"+ data_name + '_' + model_name + '_' + loss_name +\
                        '_eval_test_percent.csv', index = False)
            pd.DataFrame(eval_val_p, columns = eval_name[data_name])\
                .to_csv("1_main/output/"+ data_name + '_' + model_name + '_' + loss_name +\
                        '_eval_val_percent.csv', index = False)
            pd.DataFrame(proxy_val_p, columns = proxy_name)\
                .to_csv("1_main/output/"+ data_name + '_' + model_name + '_' + loss_name +\
                        '_proxy_val_percent.csv', index = False)
            
