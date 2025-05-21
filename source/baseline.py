import torch
import pickle
import os
import tqdm
import numpy as np
import pandas as pd
import warnings
import gurobipy as gp
import recsys_metrics
warnings.filterwarnings('ignore')

from source.model import PMF, NCF
from source.metric import \
    rmse, precision, recall, f_score, ndcg, \
    drmse, dprecision, drecall, df_score, dndcg, topic_cover
from source.influencer import influencer
from source.trainer import trainer
from source.loader import loader
from source.experiment import experiment

# %% Raw
class baseline_raw():
    
    # init function
    def __init__(self, data_name, model_name, config_model, config_train, k = 10):
        
        # init config model
        factor_num = config_model['factor_num']
        num_layers = config_model['num_layers']
        dropout = config_model['dropout']
        lambda_param = config_model['lambda_param']
        
        # init exp
        exp = experiment(data_name = data_name, model_name = model_name, loss_name = 'A',
                         device = 'cuda', contrast = 'bpr',
                         factor_num = factor_num, num_layers = num_layers, dropout = dropout,
                         r = 0.05, beta = 1,
                         lambda_param = lambda_param, lambda_F = 0, lambda_D = 0)
        self.exp = exp

        # fit
        self.exp.prepare_model()
        self.exp.fit(config = config_train)
        self.exp.evaluate(k = k)

# %% Ekstrand et al.
class baseline_ekstrand():
    
    # init function
    def __init__(self, data_name, model_name, config_model, config_train, k = 10):
        
        # init config model
        factor_num = config_model['factor_num']
        num_layers = config_model['num_layers']
        dropout = config_model['dropout']
        lambda_param = config_model['lambda_param']
        
        # init exp
        exp = experiment(data_name = data_name, model_name = model_name, loss_name = 'A',
                         device = 'cuda', contrast = 'bpr',
                         factor_num = factor_num, num_layers = num_layers, dropout = dropout,
                         r = 0.05, beta = 1,
                         lambda_param = lambda_param, lambda_F = 0, lambda_D = 0)
        self.exp = exp
        
        # preprocess data
        self.train_gender = self.exp.user_gender[self.exp.train[:, 0], 1]
        male_num = self.train_gender.sum().item()
        female_num = self.train_gender.shape[0] - male_num
        if male_num > female_num:
            self.exp.train = torch.cat((self.exp.train, self.exp.train[self.train_gender == 0, :]
                                    [torch.randint(high = female_num, size = (male_num - female_num,))]))
        else:
            self.exp.train = torch.cat((self.exp.train, self.exp.train[self.train_gender == 1, :]
                                    [torch.randint(high = male_num, size = (female_num - male_num,))]))
            
        # fit
        self.exp.prepare_model()
        self.exp.fit(config = config_train)
        self.exp.evaluate(k = k)
        
# %% Beutel et al.
class baseline_beutel():
    
    # init function
    def __init__(self, data_name, model_name, config_model, config_train, k = 10):
        
        # init config model
        factor_num = config_model['factor_num']
        num_layers = config_model['num_layers']
        dropout = config_model['dropout']
        lambda_param = config_model['lambda_param']
        
        # init exp
        exp = experiment(data_name = data_name, model_name = model_name, loss_name = 'A',
                         device = 'cuda', contrast = 'bpr',
                         factor_num = factor_num, num_layers = num_layers, dropout = dropout,
                         r = 0.05, beta = 1,
                         lambda_param = lambda_param, lambda_F = 0, lambda_D = 0)
        self.exp = exp
        self.exp.trainer.if_beutel = True
            
        # fit
        self.exp.prepare_model()
        self.exp.fit(config = config_train)
        self.exp.evaluate(k = k)        
        
# %% Zhang et al.
class baseline_zhang():
    
    # init function
    def __init__(self, data_name, model_name, config_model, config_train, r1 = 0.1, r2 = 0.05, k = 10):
        
        # init config model
        factor_num = config_model['factor_num']
        num_layers = config_model['num_layers']
        dropout = config_model['dropout']
        lambda_param = config_model['lambda_param']
        
        # init exp
        exp = experiment(data_name = data_name, model_name = model_name, loss_name = 'A',
                         device = 'cuda', contrast = 'bpr',
                         factor_num = factor_num, num_layers = num_layers, dropout = dropout,
                         r = 0.05, beta = 1,
                         lambda_param = lambda_param, lambda_F = 0, lambda_D = 0)
        self.exp = exp
        
        # preprocess data
        tags = set()
        for item, tag in self.exp.item_tag.items():
            tags = tags.union(set(tag))
        tags = list(tags)
        
        item_pop = torch.zeros((self.exp.item_num), device = 'cuda')
        for item in range(self.exp.item_num):
            item_pop[item] = (self.exp.train[:, 1] == item).sum().item()
        
        tag_item = torch.zeros((len(tags), self.exp.item_num), device = 'cuda')
        for item, tag in self.exp.item_tag.items():
            tags_ind = [tags.index(t) for t in self.exp.item_tag[item]]
            tag_item[tags_ind, item] = item_pop[item]
        
        tag_pop = tag_item.sum(1)
        tag_reserve = torch.topk(tag_pop, k = min(200, len(tags)))[1]
        tags = tag_reserve.tolist()
        tag_item = tag_item[tag_reserve, :]
        
        tag_item = tag_item/(tag_item.sum(dim = 1).reshape((-1, 1)))
        tag_item = torch.nan_to_num(tag_item, 0)
        
        tag_sim = torch.matmul(tag_item, tag_item.T)/\
            tag_item.norm(p=2, dim=1).reshape((-1, 1))/\
                tag_item.norm(p=2, dim=1).reshape((1, -1))
        tag_sim = torch.nan_to_num(tag_sim, 0)
        tag_sim = tag_sim.fill_diagonal_(1)
        
        user_tag = torch.zeros((self.exp.user_num, len(tags)), device = 'cuda')
        for user in range(self.exp.user_num):
            item_slice = self.exp.train[self.exp.train[:, 0]==user, 1]
            tag_slice = torch.where(tag_item[:, item_slice] > 0, 1, 0).sum(dim=1)
            user_tag[user, :] = tag_slice
        user_tag = user_tag/(user_tag.sum(dim = 1).reshape((-1, 1)))
        user_tag_hat = torch.matmul(user_tag, tag_sim)/(tag_sim.sum(dim=0).reshape((1, -1)))
        
        k_list = torch.clamp(torch.ceil((1 + r1)*(user_tag > 0).sum(dim=1)), min = 0, max = len(tags)).int().tolist()
        user_tag_hat_rank = user_tag_hat.argsort(dim=1, descending=True)
        for i, user in enumerate(range(self.exp.user_num)):
            k_ = k_list[i]
            user_tag_hat[user, user_tag_hat_rank[user, k_:]] = 0
        user_tag_hat = user_tag_hat/(user_tag_hat.sum(dim=1).reshape((-1, 1)))
        
        user_tag_tensor = user_tag_hat.unsqueeze(-1).cpu()
        tag_item_tensor = tag_item.unsqueeze(0).cpu()
        user_tag_item = (1 - user_tag_tensor) >= tag_item_tensor
        user_tag_item = user_tag_item.int()
        print('User-tag-item tensor construction complete. Now excluding existing interactions...')
        
        for r in range(self.exp.train.shape[0]):
            user, item = self.exp.train[r, 0], self.exp.train[r, 1]
            user_tag_item[user, :, item] = 0
        
        tag_explore_num = torch.clamp(torch.ceil((1 + r1)*(user_tag > 0).sum(dim=1)), min = 0, max = len(tags)).int()
        count = torch.ceil(r2 * (torch.unique(self.exp.train[:, 0], return_counts = True)[1].to('cuda'))/tag_explore_num)
        count = count.cpu()
        init = True
        for user in range(self.exp.user_num):
            if int(count[user])==0:
                pass
            else:
                tag_explore_ind = torch.topk(user_tag_hat[user, :], k = tag_explore_num[user])[1]
                for t in range(tag_explore_num[user]):
                    explore_tag = tag_explore_ind[t]
                    candidates = user_tag_item[user, explore_tag, :].nonzero().reshape((-1))
                    items = candidates[torch.randperm(candidates.shape[0])[:int(count[user].item())]]
                    interactions = torch.concat((torch.tensor(user).repeat(int(count[user].item()), 1), 
                                                  items.reshape((-1, 1)).to('cpu')), dim = 1)
                    if data_name == 'movielens':
                        ratings = torch.randint(low = 1, high = 5, size = (int(count[user].item()), 1))
                        interactions = torch.concat((interactions, ratings), dim = 1)
                        
                    if init:
                        add_interactions = interactions
                        init = False
                    else:
                        add_interactions = torch.concat((add_interactions, interactions))
                    
        add_interactions = add_interactions[torch.randperm(add_interactions.shape[0]).int()[\
            :int(min(add_interactions.shape[0], r2 * self.exp.train.shape[0]))], :]
        print(add_interactions.shape)
        self.exp.train = torch.concat((self.exp.train, add_interactions))
            
        # fit
        self.exp.prepare_model()
        self.exp.fit(config = config_train)
        self.exp.evaluate(k = k)
        
# %% Ziegler et al.
class baseline_ziegler():
    
    # init function
    def __init__(self, data_name, model_name, config_model, config_train, k = 10, l = 20, must_keep = 7, theta = 0.5):
        
        # init config model
        factor_num = config_model['factor_num']
        num_layers = config_model['num_layers']
        dropout = config_model['dropout']
        lambda_param = config_model['lambda_param']
        
        # init exp
        exp = experiment(data_name = data_name, model_name = model_name, loss_name = 'A',
                         device = 'cuda', contrast = 'bpr',
                         factor_num = factor_num, num_layers = num_layers, dropout = dropout,
                         r = 0.05, beta = 1,
                         lambda_param = lambda_param, lambda_F = 0, lambda_D = 0)
        self.exp = exp

        # fit
        self.exp.prepare_model()
        self.exp.fit(config = config_train)
        
        # postprocessing
        ## generating new recommendations
        model = self.exp.model
        user_unknown_mat = self.exp.user_unknown_test_mat
        item_tag = self.exp.item_tag
        total_eval = user_unknown_mat.shape[1]
        
        user = torch.unique(self.exp.test[:, 0])
        total_eval = user_unknown_mat.shape[1]
        interaction_all = torch.cat((user.repeat(total_eval, 1).T.reshape((-1, 1)), \
                                     user_unknown_mat[user, :].reshape((-1, 1))), dim = 1)
        batch_size, pred_all = 8192, torch.zeros((interaction_all.shape[0]))
        for i in range(int(interaction_all.shape[0]/batch_size)+1):
            ind = torch.arange(start = i*batch_size,
                               end = min((i+1)*batch_size, interaction_all.shape[0]))
            pred_all[ind] = model(interaction_all[ind, 0].to('cuda'), 
                                  interaction_all[ind, 1].to('cuda')).detach().to('cpu')
        pred_mat = pred_all.reshape((user.shape[0], total_eval))
        self.pred_mat_old = pred_mat
        
        new_rec = torch.zeros((user_unknown_mat.shape[0], k))
        for user in range(user_unknown_mat.shape[0]):
            topl_item = (user_unknown_mat[user, :][torch.topk(pred_mat[user, :], l)[1]])
            topl_item_list = topl_item.tolist()
            topk_item = (user_unknown_mat[user, :][torch.topk(pred_mat[user, :], k)[1]])
            topk_item_list = topk_item.tolist()
            
            topic_covered = set()
            for t in range(must_keep):
                new_rec[user, t] = topk_item_list[t]
                topic_covered = topic_covered.union(set(item_tag[topk_item_list[t]]))
            
            for z in range(must_keep, k):
                
                topl_item_sim = []
                for item in topl_item_list:
                    topl_item_sim.append(len(set(item_tag[item]).intersection(topic_covered))/
                                         len(set(item_tag[item]).union(topic_covered)))
                rank_pos = torch.argsort(torch.tensor(topl_item_sim)).tolist()
                topdissim_item_list = [topl_item_list[rank_pos[j]] for j in range(len(topl_item_list))]
                w_list = []
                for j, item in enumerate(topl_item_list):
                    w_list.append((1-theta)*j + theta*topdissim_item_list.index(item))
                
                topic_covered = topic_covered.union(set(item_tag[topl_item_list[torch.argmin(torch.tensor(w_list))]]))
                new_rec[user, z] = topl_item_list[torch.argmin(torch.tensor(w_list))]
                topl_item_list.remove(topl_item_list[torch.argmin(torch.tensor(w_list))])
                
        print('Reranking complete. Now constructing target and prediction matrix...')
                
        ## compose target_mat
        user_unknown_cut = self.exp.user_unknown_test_cut
        target_mat = torch.zeros(user_unknown_mat.shape)
        for i in range(user_unknown_mat.shape[0]):
            target_mat[i, 0:user_unknown_cut[i]] = 1
                
        ## compose pred_mat
        pred_mat = torch.zeros(target_mat.shape)
        for user in range(pred_mat.shape[0]):
            for l in range(k):
                pred_mat[user, user_unknown_mat[user, :].tolist().index(new_rec[user, l])] = 1 - l/k
                
        self.pred_mat_new = pred_mat
                
        ## compute topic cover
        topic_cover_list = torch.zeros((new_rec.shape[0]))
        for user in range(new_rec.shape[0]):
            cover = set()
            for item in new_rec[user, :].tolist():
                cover = cover.union(set(item_tag[item]))
            topic_cover_list[user] = len(cover)
        topic_cover_value = topic_cover_list.mean().item()
        print('Now evaluating...')
        
        ## preparing
        user_gender = self.exp.user_gender
        male_ind, female_ind = user_gender[self.exp.test[:, 0], 1] == 1, user_gender[self.exp.test[:, 0], 1] == 0
        male_user, female_user = user_gender[:, 1] == 1, user_gender[:, 1] == 0
        
        ## evaluate others for explicit feedback
        if data_name == 'movielens':
            user, item, rating = \
                self.exp.test[:, 0], \
                self.exp.test[:, 1], \
                self.exp.test[:, 2]
            pred_value = model(user.to('cuda'), item.to('cuda')).detach().to('cpu')
            pred_value_male, rating_male = pred_value[male_ind], rating[male_ind]
            pred_value_female, rating_female = pred_value[female_ind], rating[female_ind] 
            
            rmse_value = torch.sqrt(torch.mean((pred_value - rating)**2)).detach().item()
            drmse_value = abs(torch.sqrt(torch.mean((pred_value_male - rating_male)**2)).detach().item() - \
                        torch.sqrt(torch.mean((pred_value_female - rating_female)**2)).detach().item())
            self.exp.eval_test = [rmse_value, drmse_value, topic_cover_value]
                
        elif data_name == 'lastfm':
            precision_value = recsys_metrics.precision(pred_mat, target_mat, k = k).item()
            recall_value = recsys_metrics.recall(pred_mat, target_mat, k = k).item()
            f_score_value = 2/(1/(precision_value+1e-6) + 1/(recall_value+1e-6))
            ndcg_value = recsys_metrics.normalized_dcg(pred_mat, target_mat, k = k).item()
            pred_mat_male, target_mat_male = pred_mat[male_user, :], target_mat[male_user, :]
            pred_mat_female, target_mat_female = pred_mat[female_user, :], target_mat[female_user, :]
            dprecision_value = abs(recsys_metrics.precision(pred_mat_male, target_mat_male, k = k).item() - \
                             recsys_metrics.precision(pred_mat_female, target_mat_female, k = k).item())
            drecall_value = abs(recsys_metrics.recall(pred_mat_male, target_mat_male, k = k).item() - \
                          recsys_metrics.recall(pred_mat_female, target_mat_female, k = k).item())
            df_score_value = abs(2/(1/(recsys_metrics.precision(pred_mat_male, target_mat_male, k = k).item()+1e-6) + \
                                    1/(recsys_metrics.recall(pred_mat_male, target_mat_male, k = k).item()+1e-6)) - \
                                 2/(1/(recsys_metrics.precision(pred_mat_female, target_mat_female, k = k).item()+1e-6) + \
                                    1/(recsys_metrics.recall(pred_mat_female, target_mat_female, k = k).item()+1e-6)))
            dndcg_value = abs(recsys_metrics.normalized_dcg(pred_mat_male, target_mat_male, k = k).item() - \
                        recsys_metrics.normalized_dcg(pred_mat_female, target_mat_female, k = k).item())
            self.exp.eval_test = [precision_value, recall_value, f_score_value, ndcg_value,
                                  dprecision_value, drecall_value, df_score_value, dndcg_value, topic_cover_value]
            
        
        
        
