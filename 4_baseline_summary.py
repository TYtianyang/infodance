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
pd.set_option('display.max_columns', None)
            
# %% settings
data_names = ['movielens', 'lastfm']
model_names = ['PMF', 'NCF']

# %% table & plot   
baseline_names = os.listdir('4_baseline/output/')
for data_name in data_names:
    for model_name in model_names:
        if data_name == 'movielens':
            panel = pd.DataFrame(np.zeros((len(baseline_names), 3)), columns = ['RMSE', 'dRMSE', 'Topic Cover'],
                                 index = baseline_names)
            for baseline_name in baseline_names:
                with open('4_baseline/output/'+baseline_name, 'rb') as f:
                    baseline_performance = pickle.load(f)
                    panel.loc[baseline_name, 'RMSE'] = baseline_performance[data_name][model_name][0]
                    panel.loc[baseline_name, 'dRMSE'] = baseline_performance[data_name][model_name][1]
                    panel.loc[baseline_name, 'Topic Cover'] = baseline_performance[data_name][model_name][2]
            
        else:
            panel = pd.DataFrame(np.zeros((len(baseline_names), 5)), columns = ['Precision', 'NDCG',
                                                                                'dPrecision', 'dNDCG', 'Topic Cover'],
                                 index = baseline_names)
            for baseline_name in baseline_names:
                with open('4_baseline/output/'+baseline_name, 'rb') as f:
                    baseline_performance = pickle.load(f)
                    panel.loc[baseline_name, 'Precision'] = baseline_performance[data_name][model_name][0]
                    panel.loc[baseline_name, 'NDCG'] = baseline_performance[data_name][model_name][3]
                    panel.loc[baseline_name, 'dPrecision'] = baseline_performance[data_name][model_name][4]
                    panel.loc[baseline_name, 'dNDCG'] = baseline_performance[data_name][model_name][7]
                    panel.loc[baseline_name, 'Topic Cover'] = baseline_performance[data_name][model_name][8]
        
        panel.to_csv('4_baseline/table/' + data_name + '_' + model_name + '.csv')
