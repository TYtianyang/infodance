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
data_name = input()
print('Model name? (e.g. PMF, NCF)')
model_name = input()
print('Loss name? (e.g. A, A+F+D)')
loss_name = input()

print('what k to evaluate (e.g. 10)')
k = input()

print('List: augment ratio per step. (e.g. 0.01)')
r_list = input().replace(' ','').split(',')
print('List: augment steps. (e.g. 10)')
t_list = input().replace(' ','').split(',')
print('List: augment thres (e.g. -1, 0, -1)')
thres_list = input().replace(' ','').split(';')

# %% unchange param
config_model = {'movielens': {'PMF': {'A': {'factor_num': 16,
                                            'num_layers': 0,
                                            'dropout': 0,
                                            'lambda_param': 0.005, 
                                            'lambda_F': 0,
                                            'lambda_D': 0},
                                      'A+F+D': {'factor_num': 16,
                                                'num_layers': 0,
                                                'dropout': 0,
                                                'lambda_param': 0.005,
                                                'lambda_F': 0.0001, 
                                                'lambda_D': 0.0001}}, 
                              'NCF': {'A': {'factor_num': 16,
                                            'num_layers': 2,
                                            'dropout': 0.2,
                                            'lambda_param': 0.0025,
                                            'lambda_F': 0,
                                            'lambda_D': 0},
                                      'A+F+D': {'factor_num': 16,
                                                'num_layers': 2,
                                                'dropout': 0.2,
                                                'lambda_param': 0.0025,
                                                'lambda_F': 0.0001, 
                                                'lambda_D': 0.01}}}, 
                'lastfm': {'PMF': {'A': {'factor_num': 32,
                                         'num_layers': 0,
                                         'dropout': 0,
                                         'lambda_param': 0,
                                         'lambda_F': 0,
                                         'lambda_D': 0},
                                   'A+F+D': {'factor_num': 32,
                                             'num_layers': 0,
                                             'dropout': 0,
                                             'lambda_param': 0,
                                             'lambda_F': 0.0001, 
                                             'lambda_D': 0.1}}, 
                           'NCF': {'A': {'factor_num': 16,  
                                         'num_layers': 2, 
                                         'dropout': 0.2, 
                                         'lambda_param': 0, 
                                         'lambda_F': 0,
                                         'lambda_D': 0},
                                   'A+F+D': {'factor_num': 16, 
                                             'num_layers': 2, 
                                             'dropout': 0.2, 
                                             'lambda_param': 0, 
                                             'lambda_F': 0.01, 
                                             'lambda_D': 0.1}}}} 

config_initial_train = {'lr': 1e-2,
                        'epochs': 10000,
                        'gamma': 0.5,
                        'batch_size': 32768,
                        'patience': 1000,
                        'verbose': True}
config_dynamic_train = {'lr': 1e-4,
                        'epochs': 1000,
                        'gamma': 0.5,
                        'batch_size': 32768,
                        'patience': 200,
                        'verbose': True}
config_initial_influence = {'batch_size_first': 32768,
                            'batch_size_second': 32768,
                            'batch_size_third': 1248,
                            'epochs_first': 1000,
                            'epochs_second': 1000,
                            'epochs_check': 100,
                            'lr_second': 1,
                            'warmup_second': False}
config_dynamic_influence = {'batch_size_first': 32768,
                            'batch_size_second': 32768,
                            'batch_size_third': 1248,
                            'epochs_first': 1000,
                            'epochs_second': 100,
                            'epochs_check': 100,
                            'lr_second': 1,
                            'warmup_second': True}
if model_name == 'NCF':
    print('Initial second epochs. (e.g. 10000)')
    config_initial_influence['epochs_second'] = int(input())
    print('Initial second lr. (e.g. 0.01)')
    config_initial_influence['lr_second'] = float(input())
    print('Dynamic second epochs. (e.g. 1000)')
    config_dynamic_influence['epochs_second'] = int(input())
    print('Dynamic second lr. (e.g. 0.01)')
    config_dynamic_influence['lr_second'] = float(input())
    print('Warmup second? (e.g. True)')
    config_dynamic_influence['warmup_second'] = (input()=='True')

if data_name == 'lastfm':
    print('What contrast are you gonna use? (e.g. rank, bpr)')
    contrast = input()
else:
    contrast = 'rank'
    
# %% main
# prepare
factor_num = config_model[data_name][model_name][loss_name]['factor_num']
num_layers = config_model[data_name][model_name][loss_name]['num_layers']
dropout = config_model[data_name][model_name][loss_name]['dropout']
lambda_param = config_model[data_name][model_name][loss_name]['lambda_param']
lambda_F = config_model[data_name][model_name][loss_name]['lambda_F']
lambda_D = config_model[data_name][model_name][loss_name]['lambda_D']
k = int(k)

for r_c in r_list:
    for t_c in t_list:
        for thres_c in thres_list:
            r = float(r_c)
            t = int(t_c)
            thres = thres_c.replace(' ','').split(',')
            thres = [float(item) for item in thres]
            
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
            
            # plot
            proxy_name = ['A', 'F', 'D']
            if data_name == 'movielens':
                eval_name_base = ['rmse', 'drmse', 'topic_cover']
                eval_name = ['rmse', 'drmse', 'topic_cover']
            else:
                eval_name_base = ['precision', 'recall', 'f_score', 'ndcg', 
                                  'dprecision', 'drecall', 'df_score', 'dndcg', 'topic_cover']
                eval_name = ['precision', 'ndcg', 'dprecision', 'dndcg', 'topic_cover']
            eval_test, eval_val, proxy_val = \
                np.array(eval_test_list), \
                np.array(eval_val_list), \
                np.array([item.tolist() for item in proxy_val_list])
            eval_test, eval_val, proxy_val = \
                eval_test/eval_test[0, :], \
                eval_val/eval_val[0, :], \
                proxy_val/proxy_val[0, :]
            eval_test, eval_val = \
                eval_test[:, [eval_name_base.index(item) for item in eval_name]], \
                eval_val[:, [eval_name_base.index(item) for item in eval_name]]
            fig, axs = plt.subplots(1, 3, figsize = (15, 4))
            x = np.arange(t+1)
            for i in range(len(eval_name)):
                axs[0].plot(x, eval_test[:, i], label = eval_name[i])
            axs[0].set_title('Eval Test')
            axs[0].legend()
            for i in range(len(eval_name)):
                axs[1].plot(x, eval_val[:, i], label = eval_name[i])
            axs[1].set_title('Eval Val')
            axs[1].legend()
            for i in range(len(proxy_name)):
                axs[2].plot(x, proxy_val[:, i], label = proxy_name[i])
            axs[2].set_title('Proxy Val')
            axs[2].legend()
            plt.savefig("tune_influence_fig/" + data_name + '_' + model_name + '_' + loss_name + '_'
                        + r_c + '_' + t_c + '_' + thres_c + '.png', dpi=150)


