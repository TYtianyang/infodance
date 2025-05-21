import torch
import pickle
import os
import tqdm
import numpy as np
import pandas as pd
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('/home/u9/tianyangxie/Documents/cf')
from source.experiment import experiment
pd.set_option('display.max_columns', None)
            
# %% settings
data_names = ['movielens', 'lastfm']
model_names = ['PMF', 'NCF']
loss_names = ['A', 'A+F+D']

# %% get stop_t
patience = 5
stop_t_dict = {'movielens': {'PMF': {'A': 0, 'A+F+D': 0},
                             'NCF': {'A': 0, 'A+F+D': 0}},
               'lastfm': {'PMF': {'A': 0, 'A+F+D': 0},
                          'NCF': {'A': 0, 'A+F+D': 0}}}

for data_name in data_names:
    for model_name in model_names:
        for loss_name in loss_names:
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
                    
            # record
            stop_t_dict[data_name][model_name][loss_name] = stop_t
            
# %% 1: main fairness results
for data_name in data_names:
    # create table
    if data_name == 'movielens':
        panel = pd.DataFrame(np.zeros((6, 4)), columns = ['RMSE Female A', 'RMSE Male A', 'RMSE Female A+F+D', 'RMSE Male A+F+D'],
                             index = ['PMF before', 'PMF after', 'PMF diff',
                                      'NCF before', 'NCF after', 'NCF diff'])
    elif data_name == 'lastfm':
        panel = pd.DataFrame(np.zeros((6, 8)), columns = ['Precision Female A', 'Precision Male A', 'NDCG Female A', 'NDCG Male A',
                                                          'Precision Female A+F+D', 'Precision Male A+F+D', 'NDCG Female A+F+D', 'NDCG Male A+F+D'],
                             index = ['PMF before', 'PMF after', 'PMF diff',
                                      'NCF before', 'NCF after', 'NCF diff'])
        
    for model_name in model_names:
        for loss_name in loss_names:
            
            # read the data
            eval_table = pd.read_csv('2_inference/misc/'+data_name+'_'+model_name+'_'+loss_name+'gendereval.csv')
            
            # fill in panel
            if loss_name == 'A':
                panel.loc[model_name + ' before', panel.columns[0:int(panel.shape[1]/2)]] = np.array(eval_table.iloc[0, :])
                panel.loc[model_name + ' after', panel.columns[0:int(panel.shape[1]/2)]] = \
                    np.array(eval_table.iloc[stop_t_dict[data_name][model_name][loss_name], :])
                panel.loc[model_name + ' diff', panel.columns[0:int(panel.shape[1]/2)]] = \
                    panel.loc[model_name + ' after', panel.columns[0:int(panel.shape[1]/2)]] - \
                    panel.loc[model_name + ' before', panel.columns[0:int(panel.shape[1]/2)]]
            if loss_name == 'A+F+D':
                panel.loc[model_name + ' before', panel.columns[int(panel.shape[1]/2):]] = np.array(eval_table.iloc[0, :])
                panel.loc[model_name + ' after', panel.columns[int(panel.shape[1]/2):]] = \
                    np.array(eval_table.iloc[stop_t_dict[data_name][model_name][loss_name], :])
                panel.loc[model_name + ' diff', panel.columns[int(panel.shape[1]/2):]] = \
                    panel.loc[model_name + ' after', panel.columns[int(panel.shape[1]/2):]] - \
                    panel.loc[model_name + ' before', panel.columns[int(panel.shape[1]/2):]]
                    
            # figure
            fig, axs = plt.subplots(figsize = (5, 4))
            x = np.arange(eval_table.shape[0])
            if data_name == 'movielens':
                for i in range(eval_table.shape[1]):
                    axs.plot(x, eval_table.iloc[:, i], label = ['RMSE Female', 'RMSE Male'][i])
                axs.set_title(data_name + ' ' + model_name + ' ' + loss_name)
                axs.legend()
                axs.axvline(x = stop_t_dict[data_name][model_name][loss_name], color = 'black', linestyle = 'dashed')
            elif data_name == 'lastfm':
                for i in range(eval_table.shape[1]):
                    axs.plot(x, eval_table.iloc[:, i], label = ['Precision Female', 'Precision Male', 'NDCG Female', 'NDCG Male'][i])
                axs.set_title(data_name + ' ' + model_name + ' ' + loss_name)
                axs.legend()
                axs.axvline(x = stop_t_dict[data_name][model_name][loss_name], color = 'black', linestyle = 'dashed')
            axs.set_ylabel('Value')
            axs.set_xlabel('Step')
            plt.savefig('2_inference/fig/'+data_name + '_' + model_name + '_' + loss_name + '_fairness.svg',
                        bbox_inches='tight')
            
    # save
    panel.to_csv('2_inference/table/'+data_name+'.csv', index = True)
                    
            
# %% 2: influence value presentation
for data_name in data_names:
    if data_name == 'movielens':
        thres = [0, 0, 0]
    elif data_name == 'lastfm':
        thres = [1, -1, 1]
    for model_name in model_names:
        for loss_name in loss_names:
            candidates = pd.read_csv('2_inference/augment/'+data_name+'_'+model_name+'_'+loss_name+'.csv')
            candidates = candidates[candidates['t']==0]
            candidates['Virtual Data'] = 'Discarded'
            candidates.loc[(candidates['A'] <= thres[0])&\
                           (candidates['F'] <= thres[1])&\
                           (candidates['D'] <= thres[2]), 'Virtual Data'] = 'Selected'
            plt.figure(figsize=(10, 9))
            infl = sns.pairplot(candidates[['A', 'F', 'D', 'Virtual Data']], hue = 'Virtual Data', 
                         palette = {'Discarded': sns.color_palette()[0],
                                    'Selected': sns.color_palette()[1]},
                         diag_kind = 'kde')
            infl.fig.suptitle(data_name + ' ' + model_name + ' ' + loss_name, y = 1)
            plt.savefig('2_inference/fig/'+data_name + '_' + model_name + '_' + loss_name + '_influence.svg',
                        bbox_inches='tight')

# %% 3: density
lb, ub = 0, 0.03
for data_name in data_names:
    if data_name == 'movielens':
        thres = [0, 0, 0]
        path = 'data/ml-1m/'
    elif data_name == 'lastfm':
        thres = [1, -1, 1]
        path = 'data/lastfm-dataset-1K/'
        
    for model_name in model_names:
        for loss_name in loss_names:
            # prepare the data
            candidates = pd.read_csv('2_inference/augment/'+data_name+'_'+model_name+'_'+loss_name+'.csv')
            candidates = candidates.loc[(candidates['A'] <= thres[0])&\
                                        (candidates['F'] <= thres[1])&\
                                        (candidates['D'] <= thres[2]), ['t', 'user', 'item']]
            candidates['t'] += 1
            train = torch.load(path + 'train.pt')
            train = pd.DataFrame(train.numpy(), columns = ['user', 'item', 'rating'][0:(train.shape[1])])
            train['t'] = 0
            train = train[['t', 'user', 'item']]
            data = pd.concat((train, candidates), ignore_index = True)
            data = data.astype(int)
            user_gender = torch.load(path + 'user_gender.pt')
            data['gender'] = user_gender.numpy()[data['user'], 1]
            with open(path + 'item_tag.pickle', 'rb') as f:
                item_tag = pickle.load(f)
            i_list = [0, stop_t_dict[data_name][model_name][loss_name]]
                
            # user, item sparsity change plot
            fig, axs = plt.subplots(2, 2, figsize = (8, 7))
            for i, t in enumerate(i_list):
                sub_data = data[data['t'] <= t]
                user_freq = sub_data.groupby(by = 'user').count()['t']/(sub_data['item']).max()
                user_freq = user_freq[user_freq <= ub]
                kde = stats.gaussian_kde(user_freq)
                xx = np.linspace(lb, ub, 100)
                axs[i, 0].hist(user_freq, density = False, bins = 100, color = 'gray', fill = True)
                axs[i, 0].plot(xx, kde(xx), color = 'red')
                axs[i, 0].set_xlim([lb, ub])
                if i == 0:
                    ylim = axs[i, 0].get_ylim()
                else:
                    axs[i, 0].set_ylim(ylim)
                axs[i, 0].set_ylabel('t = '+str(t), rotation=90, size='large')
                mode = xx[np.argmax(kde(xx))]
                axs[i, 0].axvline(x = mode, color = 'blue', linestyle = 'dashed')
            axs[0, 0].set_title('User Rating Frequency')
            
            for i, t in enumerate(i_list):
                sub_data = data[data['t'] <= t]
                item_freq = sub_data.groupby(by = 'item').count()['t']/(sub_data['user']).max()
                item_freq = item_freq[item_freq <= ub]
                kde = stats.gaussian_kde(item_freq)
                xx = np.linspace(lb, ub, 100)
                axs[i, 1].hist(item_freq, density = False, bins = 100, color = 'gray', fill = True)
                axs[i, 1].plot(xx, kde(xx), color = 'red')
                axs[i, 1].set_xlim([lb, ub])
                if i == 0:
                    ylim = axs[i, 1].get_ylim()
                else:
                    axs[i, 1].set_ylim(ylim)
                mode = xx[np.argmax(kde(xx))]
                axs[i, 1].axvline(x = mode, color = 'blue', linestyle = 'dashed')
            axs[0, 1].set_title('Item Rating Frequency')
            fig.suptitle(data_name + ' ' + model_name + ' ' + loss_name, y = 1)
            
            # save
            plt.savefig('2_inference/fig/'+data_name + '_' + model_name + '_' + loss_name + '_density.svg',
                        bbox_inches='tight')
                
# %% 4: gender
lb, ub = 0, 0.03
for data_name in data_names:
    if data_name == 'movielens':
        thres = [0, 0, 0]
        path = 'data/ml-1m/'
    elif data_name == 'lastfm':
        thres = [1, -1, 1]
        path = 'data/lastfm-dataset-1K/'
        
    for model_name in model_names:
        for loss_name in loss_names:
            # prepare the data
            candidates = pd.read_csv('2_inference/augment/'+data_name+'_'+model_name+'_'+loss_name+'.csv')
            candidates = candidates.loc[(candidates['A'] <= thres[0])&\
                                        (candidates['F'] <= thres[1])&\
                                        (candidates['D'] <= thres[2]), ['t', 'user', 'item']]
            candidates['t'] += 1
            train = torch.load(path + 'train.pt')
            train = pd.DataFrame(train.numpy(), columns = ['user', 'item', 'rating'][0:(train.shape[1])])
            train['t'] = 0
            train = train[['t', 'user', 'item']]
            data = pd.concat((train, candidates), ignore_index = True)
            data = data.astype(int)
            user_gender = torch.load(path + 'user_gender.pt')
            data['gender'] = user_gender.numpy()[data['user'], 1]
            with open(path + 'item_tag.pickle', 'rb') as f:
                item_tag = pickle.load(f)
            i_list = [0, stop_t_dict[data_name][model_name][loss_name]]
                
            # male user, female user sparsity change
            fig, axs = plt.subplots(2, 3, figsize = (10, 6))
            male_kde, female_kde = [], []
            for i, t in enumerate(i_list):
                sub_data = data[data['t'] <= t]
                user_freq = sub_data[sub_data['gender']==1].\
                    groupby(by = 'user').count()['t']/(sub_data['item']).max()
                user_freq = user_freq[user_freq <= ub]
                kde = stats.gaussian_kde(user_freq)
                male_kde.append(kde)
                xx = np.linspace(lb, ub, 100)
                axs[i, 0].hist(user_freq, density = False, bins = 100, color = 'gray', fill = True)
                axs[i, 0].plot(xx, kde(xx), color = 'red')
                axs[i, 0].set_xlim([lb, ub])
                if i == 0:
                    ylim = axs[i, 0].get_ylim()
                else:
                    axs[i, 0].set_ylim(ylim)
                if i == 0:
                    axs[i, 0].get_xaxis().set_visible(False)
                axs[i, 0].set_ylabel('t = '+str(t), rotation=90, size='large')
                mode = xx[np.argmax(kde(xx))]
                axs[i, 0].axvline(x = mode, color = 'blue', linestyle = 'dashed')
            axs[0, 0].set_title('Male Rating Frequency')
            
            for i, t in enumerate(i_list):
                sub_data = data[data['t'] <= t]
                user_freq = sub_data[sub_data['gender']==0].\
                    groupby(by = 'user').count()['t']/(sub_data['item']).max()
                user_freq = user_freq[user_freq <= ub]
                kde = stats.gaussian_kde(user_freq)
                female_kde.append(kde)
                xx = np.linspace(lb, ub, 100)
                axs[i, 1].hist(user_freq, density = False, bins = 100, color = 'gray', fill = True)
                axs[i, 1].plot(xx, kde(xx), color = 'red')
                axs[i, 1].set_xlim([lb, ub])
                axs[i, 1].set_ylim(ylim)
                if i == 0:
                    axs[i, 1].get_xaxis().set_visible(False)
                mode = xx[np.argmax(kde(xx))]
                axs[i, 1].axvline(x = mode, color = 'blue', linestyle = 'dashed')
            axs[0, 1].set_title('Female Rating Frequency')
            
            for i, t in enumerate(i_list):
                xx = np.linspace(lb, ub, 100)
                contrast =  (female_kde[i](xx) - male_kde[i](xx))/\
                    (np.abs(female_kde[i](xx) - male_kde[i](xx)).max())
                axs[i, 2].plot(xx, contrast, color = 'red')
                axs[i, 2].set_xlim([lb, ub])
                if i == 0:
                    ylim = axs[i, 2].get_ylim()
                else:
                    axs[i, 2].set_ylim(ylim)
                if i == 0:
                    axs[i, 2].get_xaxis().set_visible(False)
                axs[i, 2].axhline(y = 0, color = 'black', linestyle = 'dashed')
            axs[0, 2].set_title('KDE Female - Male')
            fig.suptitle(data_name + ' ' + model_name + ' ' + loss_name, y = 1)
            
            # save
            plt.savefig('2_inference/fig/'+data_name + '_' + model_name + '_' + loss_name + '_gender.svg',
                        bbox_inches='tight')

# %% 5: topic cover
for data_name in data_names:
    if data_name == 'movielens':
        thres = [0, 0, 0]
        path = 'data/ml-1m/'
    elif data_name == 'lastfm':
        thres = [1, -1, 1]
        path = 'data/lastfm-dataset-1K/'
        
    for model_name in model_names:
        for loss_name in loss_names:
            # prepare the data
            candidates = pd.read_csv('2_inference/augment/'+data_name+'_'+model_name+'_'+loss_name+'.csv')
            candidates = candidates.loc[(candidates['A'] <= thres[0])&\
                                        (candidates['F'] <= thres[1])&\
                                        (candidates['D'] <= thres[2]), ['t', 'user', 'item']]
            candidates['t'] += 1
            train = torch.load(path + 'train.pt')
            train = pd.DataFrame(train.numpy(), columns = ['user', 'item', 'rating'][0:(train.shape[1])])
            train['t'] = 0
            train = train[['t', 'user', 'item']]
            data = pd.concat((train, candidates), ignore_index = True)
            data = data.astype(int)
            user_gender = torch.load(path + 'user_gender.pt')
            data['gender'] = user_gender.numpy()[data['user'], 1]
            with open(path + 'item_tag.pickle', 'rb') as f:
                item_tag = pickle.load(f)
            i_list = [0, stop_t_dict[data_name][model_name][loss_name]]

            tag_union = set()
            for i in range(len(item_tag)):
                tag_union = tag_union.union(item_tag[i])
            tag_union = list(tag_union)
            mat = torch.zeros((len(item_tag), len(tag_union)))
            for i in range(len(item_tag)):
                for j in range(len(item_tag[i])):
                    mat[i, tag_union.index(item_tag[i][j])] = 1
            item_tag_mat = mat.numpy()
                
            # exposure plot
            fig, axs = plt.subplots(2, 2, figsize = (8, 7))
            for i, t in enumerate(i_list):
                sub_data = data[data['t'] <= t]
                sub_data[[str(j) for j in range(len(tag_union))]] = item_tag_mat[sub_data['item'], :]
                user_freq = sub_data.groupby(by = 'user').sum()[[str(j) for j in range(len(tag_union))]]
                user_freq[user_freq >= 1] = 1
                user_freq = user_freq.sum(axis = 1)
                kde = stats.gaussian_kde(user_freq)
                xx = np.linspace(0, len(tag_union), 7)
                axs[i, 0].hist(user_freq, density = True, color = 'black', fill = False)
                axs[i, 0].plot(xx, kde(xx), color = 'black')
                axs[i, 0].set_xlim([0, len(tag_union)])
                if i == 0:
                    ylim = axs[i, 0].get_ylim()
                else:
                    axs[i, 0].set_ylim(ylim)
                if i == 0:
                    axs[i, 0].get_xaxis().set_visible(False)
                if data_name == 'lastfm':
                    axs[i, 0].set_xlim([0, 2000])
                axs[i, 0].set_ylabel('t = '+str(t), rotation=90, size='large')
            axs[0, 0].set_title('User Topic Coverage')
            
            for i, t in enumerate(i_list):
                sub_data = data[data['t'] <= t]
                freq = item_tag_mat[sub_data['item'], :].sum(axis = 0)
                freq = freq/freq.sum()
                axs[i, 1].bar(x = tag_union, height = freq, color = 'black', fill = False)
                if i == 0:
                    ylim = axs[i, 1].get_ylim()
                else:
                    axs[i, 1].set_ylim(ylim)
                axs[i, 1].get_xaxis().set_visible(False)
            axs[0, 1].set_title('Topic Exposure')
            fig.suptitle(data_name + ' ' + model_name + ' ' + loss_name, y = 1)
            
            # save
            plt.savefig('2_inference/fig/'+data_name + '_' + model_name + '_' + loss_name + '_topic.svg',
                        bbox_inches='tight')
                    
                    