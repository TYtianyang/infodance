import torch
import pickle
import os
import tqdm
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

os.chdir('/home/u9/tianyangxie/Documents/cf')
from source.experiment import experiment

# %% setup parameters
print('Data name? (e.g. movielens, lastfm)')
data_name = input()
print('Model name? (e.g. PMF, NCF)')
model_name = input()
print('Loss name? (e.g. A, A+F+D)')
loss_name = input()

print('List: factor number. (e.g. 32)')
factor_num_list = input().replace(' ','').split(',')
if model_name == 'NCF':
    print('List: number of layers. (e.g. 4)')
    num_layers_list = input().replace(' ','').split(',')
    print('List: drop out. (e.g. 0.2)')
    dropout_list =  input().replace(' ','').split(',')
else:
    num_layers_list = [0]
    dropout_list = [0]
print('List: augment ratio. (e.g. 0.05)')
r_list = input().replace(' ','').split(',')
print('List: lambda param. (e.g. 0.001)')
lambda_param_list = input().replace(' ','').split(',')
if loss_name == 'A+F+D':
    print('List: lambda F. (e.g. 0.01)')
    lambda_F_list = input().replace(' ','').split(',')
    print('List: lambda D. (e.g. 0.01)')
    lambda_D_list = input().replace(' ','').split(',')
else:
    lambda_F_list = [0]
    lambda_D_list = [0]

# %% special unchange param
if data_name == 'movielens' and model_name == 'NCF':
    config_influence = {'batch_size_first': 32768,
                        'batch_size_second': 32768,
                        'batch_size_third': 1248,
                        'epochs_first': 1000,
                        'epochs_second': 10000,
                        'lr_second': 0.1}
else:
    config_influence = {'batch_size_first': 32768,
                        'batch_size_second': 32768,
                        'batch_size_third': 1248,
                        'epochs_first': 1000,
                        'epochs_second': 1000,
                        'lr_second': 1}
    
# %% main
for factor_num in factor_num_list:
    for r in r_list:
        for lambda_param in lambda_param_list:
            for lambda_F in lambda_F_list:
                for lambda_D in lambda_D_list:
                    for num_layers in num_layers_list:
                        for dropout in dropout_list:
                            torch.manual_seed(2023)
                            exp = experiment(data_name = data_name, model_name = model_name, loss_name = loss_name,
                                             device = 'cuda',
                                             factor_num = int(factor_num), num_layers = int(num_layers), dropout = float(dropout),
                                             r = float(r), beta = 1,
                                             lambda_param = float(lambda_param), lambda_F = float(lambda_F), lambda_D = float(lambda_D))
                            exp.prepare_model()
                            exp.regular_train()
                            exp.evaluate()
                            exp.record(note = 'regular')
                            torch.manual_seed(2023)
                            exp.prepare_model()
                            exp.prepare_candidates()
                            exp.prepare_influence(config_influence)
                            exp.prepare_selection()
                            exp.influence_train()
                            exp.evaluate()
                            exp.record(note = 'influence')
                    



