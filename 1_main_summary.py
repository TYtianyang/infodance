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
pd.set_option('display.max_columns', None)
            
# %% settings
data_names = ['movielens', 'lastfm']
model_names = ['PMF', 'NCF']
loss_names = ['A', 'A+F', 'A+D', 'A+F+D']

# %% stopping criteria
patience = 5

# %% table & plot
for data_name in data_names:
    # create table
    if data_name == 'movielens':
        panel = pd.DataFrame(np.zeros((6, 12)), columns = ['RMSE A', 'dRMSE A', 'Topic Cover A',
                                                          'RMSE A+F', 'dRMSE A+F', 'Topic Cover A+F',
                                                          'RMSE A+D', 'dRMSE A+D', 'Topic Cover A+D',
                                                          'RMSE A+F+D', 'dRMSE A+F+D', 'Topic Cover A+F+D'],
                             index = ['PMF before', 'PMF after', 'PMF diff',
                                      'NCF before', 'NCF after', 'NCF diff'])
    elif data_name == 'lastfm':
        panel = pd.DataFrame(np.zeros((6, 20)), columns = ['Precision A', 'NDCG A', 'dPrecision A', 'dNDCG A', 'Topic Cover A',
                                                           'Precision A+F', 'NDCG A+F', 'dPrecision A+F', 'dNDCG A+F', 'Topic Cover A+F',
                                                           'Precision A+D', 'NDCG A+D', 'dPrecision A+D', 'dNDCG A+D', 'Topic Cover A+D',
                                                           'Precision A+F+D', 'NDCG A+F+D', 'dPrecision A+F+D', 'dNDCG A+F+D', 'Topic Cover A+F+D'],
                             index = ['PMF before', 'PMF after', 'PMF diff',
                                      'NCF before', 'NCF after', 'NCF diff'])
        
    for model_name in model_names:
        for loss_name in loss_names:
            
            try:
                # read the data
                eval_test_percent, eval_test_real, proxy_val_percent = \
                    pd.read_csv('1_main/output/'+data_name+'_'+model_name+'_'+loss_name+'_eval_test_percent.csv'), \
                    pd.read_csv('1_main/output/'+data_name+'_'+model_name+'_'+loss_name+'_eval_test_real.csv'), \
                    pd.read_csv('1_main/output/'+data_name+'_'+model_name+'_'+loss_name+'_proxy_val_percent.csv')
                    
                # select the stop t
                A_min, F_min, D_min = 1, 1, 1
                tol = 0
                for i in range(proxy_val_percent.shape[0]):
                    if data_name == 'movielens':
                        A, F, D = proxy_val_percent.loc[i, ['A', 'F', 'D']]
                        if sum([F >= F_min]) > 0:
                            tol += 1
                        else:
                            stop_t = i
                            tol = 0
                        A_min, F_min, D_min = min(A, A_min), min(F, F_min), min(D, D_min)
                        if tol >= patience:
                            break
                    elif data_name == 'lastfm':
                        stop_t = i
                    
                # fill table
                if data_name == 'movielens':
                    panel.loc[model_name+' before', [item + ' ' + loss_name for item in ['RMSE', 'dRMSE', 'Topic Cover']]] = \
                        np.array(eval_test_real.loc[0, ['rmse', 'drmse', 'topic_cover']])
                    panel.loc[model_name+' after', [item + ' ' + loss_name for item in ['RMSE', 'dRMSE', 'Topic Cover']]] = \
                        np.array(eval_test_real.loc[stop_t, ['rmse', 'drmse', 'topic_cover']])
                    panel.loc[model_name+' diff', [item + ' ' + loss_name for item in ['RMSE', 'dRMSE', 'Topic Cover']]] = \
                        np.array(eval_test_percent.loc[stop_t, ['rmse', 'drmse', 'topic_cover']]) - \
                        np.array(eval_test_percent.loc[0, ['rmse', 'drmse', 'topic_cover']])
                elif data_name == 'lastfm':
                    panel.loc[model_name+' before', [item + ' ' + loss_name for item in ['Precision', 'NDCG', 'dPrecision', 'dNDCG', 'Topic Cover']]] = \
                        np.array(eval_test_real.loc[0, ['precision', 'ndcg', 'dprecision', 'dndcg', 'topic_cover']])
                    panel.loc[model_name+' after', [item + ' ' + loss_name for item in ['Precision', 'NDCG', 'dPrecision', 'dNDCG', 'Topic Cover']]] = \
                        np.array(eval_test_real.loc[stop_t, ['precision', 'ndcg', 'dprecision', 'dndcg', 'topic_cover']])
                    panel.loc[model_name+' diff', [item + ' ' + loss_name for item in ['Precision', 'NDCG', 'dPrecision', 'dNDCG', 'Topic Cover']]] = \
                        np.array(eval_test_percent.loc[stop_t, ['precision', 'ndcg', 'dprecision', 'dndcg', 'topic_cover']]) - \
                        np.array(eval_test_percent.loc[0, ['precision', 'ndcg', 'dprecision', 'dndcg', 'topic_cover']])
                
                # save table
                panel.to_csv('1_main/table/'+data_name+'.csv', index = True)
                
                # figure
                fig, axs = plt.subplots(1, 2, figsize = (10, 4))
                x = np.arange(eval_test_percent.shape[0])
                if data_name == 'movielens':
                    for i, metric in enumerate(['rmse', 'drmse', 'topic_cover']):
                        axs[0].plot(x, eval_test_percent.loc[:, metric], label = ['RMSE', 'dRMSE', 'Topic Cover'][i])
                    axs[0].set_title('Evaluation Metrics on Test Set')
                    axs[0].legend()
                    axs[0].axvline(x = stop_t, color = 'black', linestyle = 'dashed')
                    for i, metric in enumerate(['A', 'F', 'D']):
                        axs[1].plot(x, proxy_val_percent.loc[:, metric], label = ['Proxy A', 'Proxy F', 'Proxy D'][i])
                    axs[1].set_title('Proxy On Validation Set')
                    axs[1].legend()
                    axs[1].axvline(x = stop_t, color = 'black', linestyle = 'dashed')
                elif data_name == 'lastfm':
                    for i, metric in enumerate(['precision', 'ndcg', 'dprecision', 'dndcg', 'topic_cover']):
                        axs[0].plot(x, eval_test_percent.loc[:, metric], label = ['Precision', 'NDCG', 'dPrecision', 'dNDCG', 'Topic Cover'][i])
                    axs[0].set_title('Evaluation Metrics on Test Set')
                    axs[0].legend()
                    axs[0].axvline(x = stop_t, color = 'black', linestyle = 'dashed')
                    for i, metric in enumerate(['A', 'F', 'D']):
                        axs[1].plot(x, proxy_val_percent.loc[:, metric], label = ['Proxy A', 'Proxy F', 'Proxy D'][i])
                    axs[1].set_title('Proxy On Validation Set')
                    axs[1].legend()
                    axs[1].axvline(x = stop_t, color = 'black', linestyle = 'dashed')
                axs[0].set_ylabel('Ratio')
                axs[0].set_xlabel('Step')
                axs[1].set_xlabel('Step')
                fig.suptitle(data_name + ' ' + model_name + ' ' + loss_name)
                plt.savefig('1_main/fig/'+data_name + '_' + model_name + '_' + loss_name + '.svg',
                            bbox_inches='tight')
            
            except:
                pass
            


