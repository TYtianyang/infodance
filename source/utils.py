import torch
import pickle
import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt

from source.model import PMF, NCF
from source.metric import \
    rmse, precision, recall, f_score, ndcg, \
    drmse, dprecision, drecall, df_score, dndcg, topic_cover
from source.config_fix import config_fix
from source.evaluation_func import \
    performance_proxy, fairness_proxy, diversity_proxy

# %% evaluate
def evaluate(file_name = 'movielens_PMF_A_0', k = 10):
    
    config_pack = config_fix('_'.join(file_name.split('_')[:-1]))
    config_model = config_pack.config_model
    
    # read the data & initialize metrics & load model
    data_name, model_name = file_name.split('_')[0:2]
    if data_name == 'movielens':
        train, val, test = \
            torch.load('data/ml-1m/train.pt'), \
            torch.load('data/ml-1m/val.pt'), \
            torch.load('data/ml-1m/test.pt')
        user_unknown_val_mat, user_unknown_val_cut, user_unknown_test_mat, user_unknown_test_cut = \
            torch.load('data/ml-1m//user_unknown_val_mat.pt'), \
            torch.load('data/ml-1m//user_unknown_val_cut.pt'), \
            torch.load('data/ml-1m//user_unknown_test_mat.pt'), \
            torch.load('data/ml-1m//user_unknown_test_cut.pt')
        user_gender = torch.load('data/ml-1m/user_gender.pt')
        with open('data/ml-1m/user_unknown_train.pickle', 'rb') as f:
            user_unknown_train = pickle.load(f)
        with open('data/ml-1m/user_unknown_val.pickle', 'rb') as f:
            user_unknown_val = pickle.load(f)
        with open('data/ml-1m/user_unknown_test.pickle', 'rb') as f:
            user_unknown_test = pickle.load(f)
        with open('data/ml-1m/item_tag.pickle', 'rb') as f:
            item_tag = pickle.load(f)
        metrics = [rmse(), drmse(), topic_cover(k=k)]
        user_num, item_num, style = 6040, 3706, 'explicit'
    elif data_name == 'lastfm':
        train, val, test = \
            torch.load('data/lastfm-dataset-1K/train.pt'), \
            torch.load('data/lastfm-dataset-1K/val.pt'), \
            torch.load('data/lastfm-dataset-1K/test.pt')
        user_unknown_val_mat, user_unknown_val_cut, user_unknown_test_mat, user_unknown_test_cut = \
            torch.load('data/lastfm-dataset-1K/user_unknown_val_mat.pt'), \
            torch.load('data/lastfm-dataset-1K/user_unknown_val_cut.pt'), \
            torch.load('data/lastfm-dataset-1K/user_unknown_test_mat.pt'), \
            torch.load('data/lastfm-dataset-1K/user_unknown_test_cut.pt')
        user_gender = torch.load('data/lastfm-dataset-1K/user_gender.pt')
        with open('data/lastfm-dataset-1K/user_unknown_train.pickle', 'rb') as f:
            user_unknown_train = pickle.load(f)
        with open('data/lastfm-dataset-1K/user_unknown_val.pickle', 'rb') as f:
            user_unknown_val = pickle.load(f)
        with open('data/lastfm-dataset-1K/user_unknown_test.pickle', 'rb') as f:
            user_unknown_test = pickle.load(f)
        with open('data/lastfm-dataset-1K/item_tag.pickle', 'rb') as f:
            item_tag = pickle.load(f)
        metrics = [precision(k=k), recall(k=k), f_score(k=k), ndcg(k=k),
                   dprecision(k=k), drecall(k=k), df_score(k=k), dndcg(k=k), topic_cover(k=k)]
        user_num, item_num, style = 867, 5001, 'implicit'
    
    if model_name == 'PMF':
        model = PMF(user_num, item_num, 
                    style = style ,
                    config = config_model)
    elif model_name == 'NCF':
        model = NCF(user_num, item_num, 
                    style = style,
                    config = config_model)
    model.load_state_dict(torch.load('checkpoint/' + file_name))
    
    # evaluate
    val_eval = [fn(model, val, 
                   user_gender = user_gender,
                   user_unknown = None,
                   user_unknown_mat = user_unknown_val_mat,
                   user_unknown_cut = user_unknown_val_cut,
                   item_tag = item_tag) for fn in metrics]
    test_eval = [fn(model, test, 
                    user_gender = user_gender,
                    user_unknown = None,
                    user_unknown_mat = user_unknown_test_mat,
                    user_unknown_cut = user_unknown_test_cut,
                    item_tag = item_tag) for fn in metrics]
    return val_eval, test_eval
    
# %% monitor
def monitor(file_name = 'movielens_PMF_A_0', 
            neg_portion = 2,
            kb = 10):
    
    config_pack = config_fix('_'.join(file_name.split('_')[:-1]))
    config_model = config_pack.config_model
    
    # read the data & initialize metrics & load model
    data_name, model_name = file_name.split('_')[0:2]
    if data_name == 'movielens':
        train, val, test = \
            torch.load('data/ml-1m/train.pt'), \
            torch.load('data/ml-1m/val.pt'), \
            torch.load('data/ml-1m/test.pt')
        user_gender = torch.load('data/ml-1m/user_gender.pt')
        with open('data/ml-1m/user_unknown_train.pickle', 'rb') as f:
            user_unknown_train = pickle.load(f)
        with open('data/ml-1m/user_unknown_val.pickle', 'rb') as f:
            user_unknown_val = pickle.load(f)
        with open('data/ml-1m/user_unknown_test.pickle', 'rb') as f:
            user_unknown_test = pickle.load(f)
        with open('data/ml-1m/item_tag.pickle', 'rb') as f:
            item_tag = pickle.load(f)
        user_num, item_num, style = 6040, 3706, 'explicit'
    elif data_name == 'lastfm':
        train, val, test = \
            torch.load('data/lastfm-dataset-1K/train.pt'), \
            torch.load('data/lastfm-dataset-1K/val.pt'), \
            torch.load('data/lastfm-dataset-1K/test.pt')
        user_gender = torch.load('data/lastfm-dataset-1K/user_gender.pt')
        with open('data/lastfm-dataset-1K/user_unknown_train.pickle', 'rb') as f:
            user_unknown_train = pickle.load(f)
        with open('data/lastfm-dataset-1K/user_unknown_val.pickle', 'rb') as f:
            user_unknown_val = pickle.load(f)
        with open('data/lastfm-dataset-1K/user_unknown_test.pickle', 'rb') as f:
            user_unknown_test = pickle.load(f)
        with open('data/lastfm-dataset-1K/item_tag.pickle', 'rb') as f:
            item_tag = pickle.load(f)
        user_num, item_num, style = 867, 5001, 'implicit'
    evaluation_funcs = [performance_proxy(style = style, neg_portion = neg_portion), 
                        fairness_proxy(style = style, neg_portion = neg_portion), 
                        diversity_proxy(style = style, kb = kb)]
        
    if model_name == 'PMF':
        model = PMF(user_num, item_num, 
                    style = style ,
                    config = config_model)
    elif model_name == 'NCF':
        model = NCF(user_num, item_num, 
                    style = style,
                    config = config_model)
    model.load_state_dict(torch.load('checkpoint/' + file_name))
    
    # evaluate
    val_moni = [fn(model, val, 
                   user_gender = user_gender,
                   user_unknown = user_unknown_train,
                   item_tag = item_tag).item() for fn in evaluation_funcs]
    return val_moni

# %% train_table (for selecting the optimal intial model)
def train_table(task_name = 'movielens_PMF_A', 
                k = 10, 
                part = 'val',
                save = True):
    
    # initialize some param
    if 'movielens' in task_name:
        colnames = ['RMSE', 'dRMSE', 'Topic Cover']
    else:
        colnames = ['Precision', 'Recall', 'F', 'NDCG',
                    'dPrecision', 'dRecall', 'dF', 'dNDCG', 'Topic Cover']
    
    # find file and record
    i = 0
    metric_list = []
    while True:
        file_name = task_name + '_' + str(i)
        try:
            val_eval, test_eval = evaluate(file_name, k)
            if part == 'val':
                metric_list.append(val_eval)
            elif part == 'test':
                metric_list.append(test_eval)
            i += 1
        except:
            break

    # return panel
    panel = pd.DataFrame(data = metric_list,
                         columns = colnames)
    if save:
        panel.to_excel('output/' + task_name + '_TT.xlsx', 
                       sheet_name = part, 
                       index = False)
    return panel

# %% augment_table (for selecting the optimal augmented model)
def augment_table(task_name = 'movielens_PMF_A_0', 
                  k = 10, 
                  part = 'test',
                  save = True,
                  last = 100):
    
    # initialize some param
    if task_name.split('_')[0] == 'movielens':
        colnames = ['RMSE', 'dRMSE', 'Topic Cover']
    else:
        colnames = ['Precision', 'Recall', 'F', 'NDCG',
                    'dPrecision', 'dRecall', 'dF', 'dNDCG', 'Topic Cover']
    
    # find file and record
    i = 0
    metric_list = []
    while True:
        file_name = task_name + '_' + str(i) + '_' + str(last)
        try:
            val_eval, test_eval = evaluate(file_name, k)
            if part == 'val':
                metric_list.append(val_eval)
            elif part == 'test':
                metric_list.append(test_eval)
            i += 1
        except:
            break

    # return panel
    panel = pd.DataFrame(data = metric_list,
                         columns = colnames)
    if save:
        panel.to_excel('output/' + '_'.join(task_name.split('_')[0:3]) + '_AT.xlsx', 
                       sheet_name = part, 
                       index = False)
    return panel

# %% augment_plot
def augment_plot(task_name = 'movielens_PMF_A_0_0', 
                 k = 10,
                 neg_portion = 2,
                 kb = 20,
                 include_figures = ['val_evaluation', 'val_metric', 'test_metric']):
    
    # initialize some param
    if task_name.split('_')[0] == 'movielens':
        metric_colnames = ['RMSE', 'dRMSE', 'Topic Cover']
    else:
        metric_colnames = ['Precision', 'Recall', 'F', 'NDCG',
                    'dPrecision', 'dRecall', 'dF', 'dNDCG', 'Topic Cover']
    evaluation_columns = ['Accuracy Proxy', 'Fairness Proxy', 'Diversity Proxy']

    # find file and record
    print('Plotting!')
    val_metric_list, val_evaluation_list, test_metric_list = [], [], []
    step_list = []
    
    val_eval, test_eval = evaluate('_'.join(task_name.split('_')[:-1]), k)
    val_moni = monitor('_'.join(task_name.split('_')[:-1]), neg_portion = neg_portion, kb = kb)
    val_metric_list.append(val_eval)
    val_evaluation_list.append(val_moni)
    test_metric_list.append(test_eval)
    step_list.append(0)
    
    file_names = [file_name for file_name in os.listdir('checkpoint/') 
                  if (task_name + '_') in file_name]
    
    for file_name in tqdm.tqdm(file_names):
        step = file_name.split('_')[-1]
        try:
            val_eval, test_eval = evaluate(file_name, k)
            val_moni = monitor(file_name, neg_portion = neg_portion, kb = kb)
            val_metric_list.append(val_eval)
            val_evaluation_list.append(val_moni)
            test_metric_list.append(test_eval)
            step_list.append(int(step))
        except:
            continue
        
    # construct panels
    panel_val_metric = pd.DataFrame(data = val_metric_list, 
                                    columns = metric_colnames,
                                    index = step_list).sort_index()
    panel_val_evaluation = pd.DataFrame(data = val_evaluation_list, 
                                        columns = evaluation_columns,
                                        index = step_list).sort_index()
    panel_test_metric = pd.DataFrame(data = test_metric_list, 
                                     columns = metric_colnames,
                                     index = step_list).sort_index()
    
    # scale function
    def scale(df):
        return (df - df.min())/(df.max() - df.min())
    
    # plots
    if 'val_evaluation' in include_figures:
        scale(panel_val_evaluation).plot()
    if 'val_metric' in include_figures:
        scale(panel_val_metric).plot()
    if 'test_metric' in include_figures:
        scale(panel_test_metric).plot()
    
    # return panels
    return panel_val_evaluation, panel_val_metric, panel_test_metric
    
# %% best_note
def best_note():
    table_names = os.listdir('output/')
    best_dict = {}
    for table_name in table_names:
        if '_TT' in table_name:
            task_name = table_name.replace('_TT.xlsx', '')
            table = pd.read_excel('output/' + table_name)
            if 'A+F+D' in task_name:
                care = 'A+F+D'
            else:
                care = 'A'
            if 'movielens' in task_name:
                rmse_fac = table['RMSE']/table['RMSE'].min()
                drmse_fac = table['dRMSE']/table['dRMSE'].min()
                topic_fac = table['Topic Cover']/table['Topic Cover'].min()
                if care == 'A':
                    best_ind = rmse_fac.argmin()
                elif care == 'A+F+D':
                    best_ind = (rmse_fac + drmse_fac - topic_fac).argmin()
                best_dict[task_name] = str(best_ind)
            elif 'lastfm' in task_name:
                ndcg_fac = table['NDCG']/table['NDCG'].min()
                dndcg_fac = table['dNDCG']/table['dNDCG'].min()
                topic_fac = table['Topic Cover']/table['Topic Cover'].min()
                if care == 'A':
                    best_ind = ndcg_fac.argmax()
                if care == 'A+F+D':
                    best_ind = (- ndcg_fac + dndcg_fac - topic_fac).argmin()
                best_dict[task_name] = str(best_ind)
    return best_dict
    
    
