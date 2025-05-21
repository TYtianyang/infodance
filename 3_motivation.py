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

# %% settings
data_names = ['movielens', 'lastfm']
model_names = ['PMF', 'NCF']
loss_names = ['A']

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

# %% 1: random sparse
suffer_list = [0.05 * i for i in range(4 + 1)]
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
            
            panel = pd.DataFrame(0, index = suffer_list, columns = eval_name[data_name])
            for j, suffer in enumerate(suffer_list):
                # run
                torch.manual_seed(2023)
                exp = experiment(data_name = data_name, model_name = model_name, loss_name = loss_name,
                                 device = 'cuda', contrast = contrast,
                                 factor_num = factor_num, num_layers = num_layers, dropout = dropout,
                                 r = r, beta = 1,
                                 lambda_param = lambda_param, lambda_F = lambda_F, lambda_D = lambda_D)
                
                perm = torch.randperm(exp.train.shape[0])[:(exp.train.shape[0]-int(exp.train.shape[0]*suffer))]
                exp.train = exp.train[perm, :]
                
                exp.prepare_model()
                exp.fit(config = config_initial_train)
                exp.evaluate(k=k)
                exp.monitor()
                panel.iloc[j, :] = exp.eval_test
                    
            # save
            panel.to_csv("3_motivation/table/"+ data_name + '_' + model_name + '_' + loss_name +\
                         '_random.csv', index = True)
                
# %% 2: gender sparse
suffer_list = [0.05 * i for i in range(4 + 1)]
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
            
            panel = pd.DataFrame(0, index = suffer_list, columns = eval_name[data_name])
            for j, suffer in enumerate(suffer_list):
                # run
                torch.manual_seed(2023)
                exp = experiment(data_name = data_name, model_name = model_name, loss_name = loss_name,
                                 device = 'cuda', contrast = contrast,
                                 factor_num = factor_num, num_layers = num_layers, dropout = dropout,
                                 r = r, beta = 1,
                                 lambda_param = lambda_param, lambda_F = lambda_F, lambda_D = lambda_D)
                
                train_gender = exp.user_gender[exp.train[:, 0], 1]
                if data_name == 'movielens':
                    ad_index = ((train_gender == 1).nonzero(as_tuple=True)[0])
                    disad_index = ((train_gender == 0).nonzero(as_tuple=True)[0])
                elif data_name == 'lastfm':
                    ad_index = ((train_gender == 0).nonzero(as_tuple=True)[0])
                    disad_index = ((train_gender == 1).nonzero(as_tuple=True)[0])
                perm = torch.randperm(disad_index.shape[0])[:(disad_index.shape[0]-int(disad_index.shape[0]*suffer))]
                exp.train = exp.train[torch.cat((ad_index, disad_index[perm])), :]
                
                exp.prepare_model()
                exp.fit(config = config_initial_train)
                exp.evaluate(k=k)
                exp.monitor()
                panel.iloc[j, :] = exp.eval_test
                    
            # save
            panel.to_csv("3_motivation/table/"+ data_name + '_' + model_name + '_' + loss_name +\
                         '_gendersparse.csv', index = True)
            
# %% 3: topic sparse
item_property = {'movielens': {}, 'lastfm': {}}
for data_name in data_names:
    if data_name == 'movielens':
        path = 'data/ml-1m/'
    elif data_name == 'lastfm':
        path = 'data/lastfm-dataset-1K/'
        
    # prepare the data
    train = torch.load(path + 'train.pt')
    train = pd.DataFrame(train.numpy(), columns = ['user', 'item', 'rating'][0:(train.shape[1])])
    train['t'] = 0
    train = train[['t', 'user', 'item']]
    data = train
    data = data.astype(int)
    with open(path + 'item_tag.pickle', 'rb') as f:
        item_tag = pickle.load(f)

    tag_union = set()
    for i in range(len(item_tag)):
        tag_union = tag_union.union(item_tag[i])
    tag_union = list(tag_union)
    mat = torch.zeros((len(item_tag), len(tag_union)))
    for i in range(len(item_tag)):
        for j in range(len(item_tag[i])):
            mat[i, tag_union.index(item_tag[i][j])] = 1
    item_tag_mat = mat.numpy()

    item_tag_sum = item_tag_mat.sum(axis = 0)
    topic_disad = np.argsort(item_tag_sum)[0:int(item_tag_sum.shape[0]*0.3)]
    item_tag_mat_disad = item_tag_mat[:, topic_disad]
    item_tag_sum_disad = item_tag_mat_disad.sum(axis = 1)
    item_property[data_name]['disad'] = np.nonzero(item_tag_sum_disad)[0]
    item_property[data_name]['ad'] = np.nonzero(item_tag_sum_disad == 0)[0]

suffer_list = [0.15 * i for i in range(4 + 1)]
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
            
            panel = pd.DataFrame(0, index = suffer_list, columns = eval_name[data_name])
            for j, suffer in enumerate(suffer_list):
                # run
                torch.manual_seed(2023)
                exp = experiment(data_name = data_name, model_name = model_name, loss_name = loss_name,
                                 device = 'cuda', contrast = contrast,
                                 factor_num = factor_num, num_layers = num_layers, dropout = dropout,
                                 r = r, beta = 1,
                                 lambda_param = lambda_param, lambda_F = lambda_F, lambda_D = lambda_D)
                
                ad_index = ((torch.isin(exp.train[:, 1], torch.from_numpy(item_property[data_name]['ad']))).nonzero(as_tuple=True)[0])
                disad_index = ((torch.isin(exp.train[:, 1], torch.from_numpy(item_property[data_name]['disad']))).nonzero(as_tuple=True)[0])
                perm = torch.randperm(disad_index.shape[0])[:(disad_index.shape[0]-int(disad_index.shape[0]*suffer))]
                exp.train = exp.train[torch.cat((ad_index, disad_index[perm])), :]
                
                exp.prepare_model()
                exp.fit(config = config_initial_train)
                exp.evaluate(k=k)
                exp.monitor()
                panel.iloc[j, :] = exp.eval_test
                    
            # save
            panel.to_csv("3_motivation/table/"+ data_name + '_' + model_name + '_' + loss_name +\
                         '_topicsparse.csv', index = True)

# %% 4: plots
for data_name in data_names:
    for model_name in model_names:
        for loss_name in loss_names:
            for style in ['random', 'gendersparse', 'topicsparse']:
                panel = pd.read_csv("3_motivation/table/"+ data_name + '_' + model_name + '_' + loss_name +\
                                    '_' + style + '.csv', index_col = 0)
                panel = panel/panel.iloc[0, :]
                
                if data_name == 'movielens':
                    plot = panel.plot(y = ['rmse', 'drmse', 'topic_cover'], 
                                      xlabel = 'Data Dropping Proportion',
                                      ylabel = 'Metrics Proportion',
                                      title = data_name + '_' + model_name + '_' + loss_name + '_' + style)
                elif data_name == 'lastfm':
                    plot = panel.plot(y = ['precision', 'ndcg', 'dprecision', 'dndcg', 'topic_cover'], 
                                      xlabel = 'Data Dropping Proportion',
                                      ylabel = 'Metrics Proportion',
                                      title = data_name + '_' + model_name + '_' + loss_name + '_' + style)
                    
                plt.savefig("3_motivation/fig/"+ data_name + '_' + model_name + '_' + loss_name +\
                            '_' + style + '.png')
                    
                    
                    
                    
                    
                    