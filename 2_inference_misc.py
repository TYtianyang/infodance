import torch
import pickle
import os
import tqdm
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

from source.experiment import experiment
from source.loader import loader
from source.metric import grmse, gprecision, gndcg

# %% setup parameters
print('List: data names? (e.g., movielens, lastfm)')
data_names = input().split(',')
print('List: model names? (e.g., PMF, NCF)')
model_names = input().split(',')
print('List: loss names? (e.g., A, A+F+D)')
loss_names = input().split(',')

k = 10
eval_name = {'movielens': ['rmse', 'drmse', 'topic_cover'],
             'lastfm': ['precision', 'recall', 'f_score', 'ndcg', 
                        'dprecision', 'drecall', 'df_score', 'dndcg', 'topic_cover']}
proxy_name = ['A', 'F', 'D']

config = {'movielens': {'PMF': {'A': {}, 'A+F+D':{}},
                        'NCF': {'A': {}, 'A+F+D':{}}},
          'lastfm': {'PMF': {'A': {}, 'A+F+D':{}},
                     'NCF': {'A': {}, 'A+F+D':{}}}}

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

# %% define misc functions
def gender_evaluate(exp, k = 10):
    if exp.style == 'explicit':
        fn = grmse()
        female_rmse, male_rmse = fn(exp.model, exp.test, 
                                    user_gender = exp.user_gender,
                                    user_unknown = None,
                                    user_unknown_mat = exp.user_unknown_test_mat,
                                    user_unknown_cut = exp.user_unknown_test_cut,
                                    item_tag = exp.item_tag)
        return pd.DataFrame({'female_rmse': female_rmse, 'male_rmse': male_rmse}, index = [0])
    elif exp.style == 'implicit':
        fn = gprecision()
        female_precision, male_precision = fn(exp.model, exp.test, 
                                              user_gender = exp.user_gender,
                                              user_unknown = None,
                                              user_unknown_mat = exp.user_unknown_test_mat,
                                              user_unknown_cut = exp.user_unknown_test_cut,
                                              item_tag = exp.item_tag)
        fn = gndcg()
        female_ndcg, male_ndcg = fn(exp.model, exp.test, 
                                    user_gender = exp.user_gender,
                                    user_unknown = None,
                                    user_unknown_mat = exp.user_unknown_test_mat,
                                    user_unknown_cut = exp.user_unknown_test_cut,
                                    item_tag = exp.item_tag)
        return pd.DataFrame({'female_precision': female_precision, 'male_precision': male_precision, 
                             'female_ndcg': female_ndcg, 'male_ndcg': male_ndcg}, index = [0])
        

# %% run the experiment
for data_name in data_names:
    if data_name == 'movielens':
        title = ['user', 'item', 'rating']
    elif data_name == 'lastfm':
        title = ['user', 'item']
    
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
            
            # read candidates
            candidates = pd.read_csv("2_inference/augment/"+ data_name + 
                                     '_' + model_name + '_' + loss_name +'.csv')
            
            # run
            torch.manual_seed(2023)
            exp = experiment(data_name = data_name, model_name = model_name, loss_name = loss_name,
                             device = 'cuda', contrast = contrast,
                             factor_num = factor_num, num_layers = num_layers, dropout = dropout,
                             r = r, beta = 1,
                             lambda_param = lambda_param, lambda_F = lambda_F, lambda_D = lambda_D)
            exp.prepare_model()
            exp.fit(config = config_initial_train)
            panel = gender_evaluate(exp, k = k)
                
            for i in range(t):
                sub_candidates = candidates[candidates['t']==i]
                sub_candidates = sub_candidates.loc[(sub_candidates['A'] < 
                                                    config[data_name][model_name][loss_name]['config_augment']['thres'][0])&
                                                    (sub_candidates['F'] < 
                                                    config[data_name][model_name][loss_name]['config_augment']['thres'][1])&
                                                    (sub_candidates['D'] < 
                                                    config[data_name][model_name][loss_name]['config_augment']['thres'][2]), title]
                sub_candidates = torch.from_numpy(np.array(sub_candidates)).long()
                exp.train = torch.cat((exp.train, sub_candidates))
                exp.loader_train = loader(exp.model, exp.train, exp.user_gender, exp.item_tag,
                                          exp.user_num, exp.item_num, style = exp.style, device = exp.device)
                exp.fit(config = config_dynamic_train)
                sub_panel = gender_evaluate(exp, k = k)
                panel = pd.concat((panel, sub_panel), ignore_index = True)
                
            # save
            panel.to_csv("2_inference/misc/"+ data_name + '_' + model_name + '_' + loss_name +\
                         'gendereval.csv', index = False)
            
