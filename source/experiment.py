import torch
import pickle
import os
import tqdm
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

os.chdir('/home/u9/tianyangxie/Documents/cf')
from source.model import PMF, NCF
from source.metric import \
    rmse, precision, recall, f_score, ndcg, \
    drmse, dprecision, drecall, df_score, dndcg, topic_cover
from source.influencer import influencer
from source.trainer import trainer
from source.loader import loader
from source.metric import \
    rmse, precision, recall, f_score, ndcg, \
    drmse, dprecision, drecall, df_score, dndcg, topic_cover

# %% experiment class
class experiment():
    
    # init function
    def __init__(self,
                 data_name, model_name, loss_name,
                 device = 'cuda', contrast = 'bpr',
                 factor_num = 128, num_layers = 4, dropout = 0.2,
                 r = 0.05, beta = 1,
                 lambda_param = 1e-3, lambda_F = 1e-2, lambda_D = 1e-2):
        
        # prepare the data
        self.device = device
        self.contrast = contrast
        self.data_name = data_name
        self.model_name = model_name
        self.loss_name = loss_name
        self.factor_num = factor_num
        self.num_layers = num_layers
        self.dropout = dropout
        self.lambda_param = lambda_param
        self.lambda_F = lambda_F
        self.lambda_D = lambda_D
        self.r = r
        self.beta = beta
        self.prepare_data()
        self.selection_size = []
        
        # set up trainer
        self.trainer = trainer(style = self.style,
                               loss_name = self.loss_name, 
                               lambda_param = self.lambda_param,
                               lambda_F = self.lambda_F, 
                               lambda_D = self.lambda_D)
        
        # set up influence function
        self.influencer = influencer(style = self.style,
                                     loss_name = self.loss_name,
                                     lambda_param = self.lambda_param,
                                     lambda_F = self.lambda_F, 
                                     lambda_D = self.lambda_D)
    
    # prepare model
    def prepare_model(self):
        if self.model_name == 'PMF':
            self.model = PMF(self.user_num, self.item_num, 
                             style = self.style,
                             factor_num = self.factor_num).to(self.device)
        elif self.model_name == 'NCF':
            self.model = NCF(self.user_num, self.item_num, 
                             style = self.style,
                             factor_num = self.factor_num,
                             num_layers = self.num_layers,
                             dropout = self.dropout).to(self.device)
        self.loader_train = loader(self.model, self.train, self.user_gender, self.item_tag,
                                   self.user_num, self.item_num, style = self.style, device = self.device,
                                   contrast = self.contrast)
        self.loader_val = loader(self.model, self.val, self.user_gender, self.item_tag,
                                 self.user_num, self.item_num, style = self.style, device = self.device,
                                 contrast = self.contrast)
    
    # prepare_data
    def prepare_data(self):
        if self.data_name == 'movielens':
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
            user_unknown_val_mat, user_unknown_val_cut, user_unknown_test_mat, user_unknown_test_cut = \
                torch.load('data/ml-1m//user_unknown_val_mat.pt'), \
                torch.load('data/ml-1m//user_unknown_val_cut.pt'), \
                torch.load('data/ml-1m//user_unknown_test_mat.pt'), \
                torch.load('data/ml-1m//user_unknown_test_cut.pt')
            self.user_num, self.item_num, self.style = \
                user_gender.shape[0], len(item_tag), 'explicit'
        elif self.data_name == 'lastfm':
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
            user_unknown_val_mat, user_unknown_val_cut, user_unknown_test_mat, user_unknown_test_cut = \
                torch.load('data/lastfm-dataset-1K/user_unknown_val_mat.pt'), \
                torch.load('data/lastfm-dataset-1K/user_unknown_val_cut.pt'), \
                torch.load('data/lastfm-dataset-1K/user_unknown_test_mat.pt'), \
                torch.load('data/lastfm-dataset-1K/user_unknown_test_cut.pt')
            self.user_num, self.item_num, self.style = \
                user_gender.shape[0], len(item_tag), 'implicit'
        self.train, self.val, self.test = train, val, test
        self.user_gender, self.user_unknown_train, self.user_unknown_val, self.user_unknown_test, self.item_tag = \
            user_gender, user_unknown_train, user_unknown_val, user_unknown_test, item_tag
        self.user_unknown_val_mat, self.user_unknown_val_cut, \
            self.user_unknown_test_mat, self.user_unknown_test_cut = \
                user_unknown_val_mat, user_unknown_val_cut, \
                    user_unknown_test_mat, user_unknown_test_cut 
        self.task_name = '_'.join([self.data_name, self.model_name, self.loss_name])
        self.augment_size = int(self.train.shape[0] * self.r)
            
    # prepare_candidates
    def prepare_candidates(self, config = {'method': 'random', # random or preselect
                                           'batch_size': 32768,
                                           'explicit_stimulus': 1}):
        
        if config['method'] == 'preselect':
            
            # setup
            batch_num = int(self.train.shape[0]/config['batch_size']) + 1
            sector_size = int(np.sqrt(self.train.shape[0]*self.r/4))
            
            # degree
            user_degree = torch.unique(self.train[:, 0], sorted = True, return_counts = True)[1]
            item_degree = torch.unique(self.train[:, 1], sorted = True, return_counts = True)[1]
            user_lower = torch.sort(user_degree, descending = False)[1][:sector_size]
            item_lower = torch.sort(item_degree, descending = False)[1][:sector_size]
            user_upper = torch.sort(user_degree, descending = True)[1][:sector_size]
            item_upper = torch.sort(item_degree, descending = True)[1][:sector_size]
            candidates_lower = torch.cat((\
                user_lower.repeat((sector_size, 1)).T.reshape((-1, 1)), \
                item_lower.repeat(sector_size).reshape((-1, 1))), dim = 1)
            candidates_upper = torch.cat((\
                user_upper.repeat((sector_size, 1)).T.reshape((-1, 1)), \
                item_upper.repeat(sector_size).reshape((-1, 1))), dim = 1)
            if self.style == 'explicit':
                candidates_lower = torch.cat((candidates_lower, 
                                              torch.tensor([config['explicit_stimulus']]).repeat(sector_size**2, 1)), dim = 1)
                candidates_upper = torch.cat((candidates_upper, 
                                              torch.tensor([config['explicit_stimulus']]).repeat(sector_size**2, 1)), dim = 1)
            candidates_degree = torch.cat((candidates_lower, candidates_upper), dim = 0)
            
            # score
            score = torch.zeros((self.train.shape[0]), device = self.device)
            for j in range(batch_num):
                ind = torch.arange(start = j * config['batch_size'],
                                   end = min(((j+1)*config['batch_size']), self.train.shape[0]))
                if self.style == 'explicit':
                    user, item, rating = self.train[ind, 0], self.train[ind, 1], self.train[ind, 2]
                    user, item, rating = user.to(self.device), item.to(self.device), rating.to(self.device)
                    pred = self.model.forward(user, item).detach()
                    loss = (pred - rating)**2
                    score[ind] = - loss
                elif self.style == 'implicit':
                    user, item = self.train[ind, 0], self.train[ind, 1]
                    user, item = user.to(self.device), item.to(self.device)
                    pred = self.model.forward(user, item).detach()
                    score[ind] = torch.log(pred + 1e-6)
            user_score = torch.zeros((self.user_num))
            item_score = torch.zeros((self.item_num))
            for user_ind in range(self.user_num):
                user_score[user_ind] = torch.mean(score[self.train[:, 0] == user_ind])
            for item_ind in range(self.item_num):
                item_score[item_ind] = torch.mean(score[self.train[:, 1] == item_ind])
            user_lower = torch.sort(user_score, descending = False)[1][:sector_size]
            item_lower = torch.sort(item_score, descending = False)[1][:sector_size]
            user_upper = torch.sort(user_score, descending = True)[1][:sector_size]
            item_upper = torch.sort(item_score, descending = True)[1][:sector_size]
            candidates_lower = torch.cat((\
                user_lower.repeat((sector_size, 1)).T.reshape((-1, 1)), \
                item_lower.repeat(sector_size).reshape((-1, 1))), dim = 1)
            candidates_upper = torch.cat((\
                user_upper.repeat((sector_size, 1)).T.reshape((-1, 1)), \
                item_upper.repeat(sector_size).reshape((-1, 1))), dim = 1)
            if self.style == 'explicit':
                candidates_lower = torch.cat((candidates_lower, 
                                              torch.tensor([config['explicit_stimulus']]).repeat(sector_size**2, 1)), dim = 1)
                candidates_upper = torch.cat((candidates_upper, 
                                              torch.tensor([config['explicit_stimulus']]).repeat(sector_size**2, 1)), dim = 1)
            candidates_score = torch.cat((candidates_lower, candidates_upper), dim = 0)
            
            # combine
            self.candidates = torch.cat((candidates_degree, candidates_score), dim = 0)
            
        elif config['method'] == 'random':
            sample_size = self.augment_size
            if self.style == 'explicit':
                self.candidates = torch.cat((torch.randint(self.user_num, size = (sample_size, 1)),
                                             torch.randint(self.item_num, size = (sample_size, 1)),
                                             torch.randint(1, 5, size = (sample_size, 1))), 
                                             dim = 1)
            elif self.style == 'implicit':
                self.candidates = torch.cat((torch.randint(self.user_num, size = (sample_size, 1)),
                                             torch.randint(self.item_num, size = (sample_size, 1))), 
                                             dim = 1)
        
    # prepare influence
    def prepare_influence(self, config = {'batch_size_first': 32768,
                                          'batch_size_second': 32768,
                                          'batch_size_third': 1248,
                                          'epochs_first': 1000,
                                          'epochs_second': 1000,
                                          'epochs_check': 1000,
                                          'lr_second': 1,
                                          'warmup_second': True}):
        self.influencer.first_component(self.loader_val, config = {'batch_size': config['batch_size_first'],
                                                                   'epochs': config['epochs_first']})
        self.influencer.second_component(self.loader_train, config = {'batch_size': config['batch_size_second'],
                                                                      'lr': config['lr_second'],
                                                                      'epochs': config['epochs_second'],
                                                                      'warmup': config['warmup_second']})
        self.influencer.check_vhps(self.loader_train, config = {'batch_size': config['batch_size_second'],
                                                                'epochs': config['epochs_check']})
        self.influencer.third_component(self.model, self.candidates, config = {'batch_size': config['batch_size_third']})
        self.influence_value = self.influencer.influence_value
    
    # prepare selection
    def prepare_selection(self, thres = [0, 0, 0]):
        self.std_influence_value = self.influence_value/self.influence_value.std()
        self.selection = self.candidates[((self.influence_value[:, 0] < thres[0]).float() +\
                                          (self.influence_value[:, 1] < thres[1]).float() +\
                                          (self.influence_value[:, 2] < thres[2]).float()) == 3, :]
        self.selection_size.append(self.selection.shape[0])
        self.train = torch.cat((self.train, self.selection))
        self.loader_train = loader(self.model, self.train, self.user_gender, self.item_tag,
                                   self.user_num, self.item_num, style = self.style, device = self.device)
        
    # fit
    def fit(self, config = {'lr': 1e-2,
                            'epochs': 10000,
                            'gamma': 0.5,
                            'batch_size': 32768,
                            'patience': 1000,
                            'verbose': True}):
        self.trainer.fit(self.loader_train, config = config)
        
    # monitor
    def monitor(self, config = {'batch_size': 32768,
                                'epochs': 1000}):
        self.proxy_val = torch.zeros((3))
        for epoch in range(config['epochs']):
            self.loader_val.generate(batch_size = config['batch_size'])
            user, pos_item, neg_item, rating = \
                self.loader_val.batch['user'], \
                self.loader_val.batch['pos_item'], \
                self.loader_val.batch['neg_item'], \
                self.loader_val.batch['rating']
            self.proxy_val[0] += self.loader_val.performance_proxy(
                user, pos_item,neg_item, rating).detach().to('cpu').item()/config['epochs']
            self.proxy_val[1] += self.loader_val.fairness_proxy(
                user, pos_item, neg_item, rating).detach().to('cpu').item()/config['epochs']
            self.proxy_val[2] += self.loader_val.diversity_proxy(
                user, pos_item, neg_item, rating).detach().to('cpu').item()/config['epochs']
                
    # evaluate
    def evaluate(self, k = 10):
        if self.style == 'explicit':
            metrics = [rmse(), drmse(), topic_cover(k=k)]
            self.metrics_title = ['rmse', 'drmse', 'topic_cover']
        elif self.style == 'implicit':
            metrics = [precision(k=k), recall(k=k), f_score(k=k), ndcg(k=k),
                       dprecision(k=k), drecall(k=k), df_score(k=k), dndcg(k=k), topic_cover(k=k)]
            self.metrics_title = ['precision', 'recall', 'f_score', 'ndcg', 
                                  'dprecision', 'drecall', 'df_score', 'dndcg', 'topic_cover']
        self.eval_val = [fn(self.model, self.val, 
                            user_gender = self.user_gender,
                            user_unknown = None,
                            user_unknown_mat = self.user_unknown_val_mat,
                            user_unknown_cut = self.user_unknown_val_cut,
                            item_tag = self.item_tag) for fn in metrics]
        self.eval_test = [fn(self.model, self.test, 
                             user_gender = self.user_gender,
                             user_unknown = None,
                             user_unknown_mat = self.user_unknown_test_mat,
                             user_unknown_cut = self.user_unknown_test_cut,
                             item_tag = self.item_tag) for fn in metrics]
        self.model = self.model.to(self.device)
    
    # record
    def record(self, note = 'regular'):
        if self.task_name not in os.listdir('tune/'):
            os.mkdir('tune/'+self.task_name)
        title = ['factor_num', 'num_layers', 'dropout', 'r', 'beta', 
                 'lambda_param', 'lambda_F', 'lambda_D'] + self.metrics_title
        
        result = pd.DataFrame(index = [0], columns = title)
        result.iloc[0, :] = [self.factor_num, self.num_layers, self.dropout, self.r, self.beta,
                             self.lambda_param, self.lambda_F, self.lambda_D] + self.eval_val
        try:
            panel = pd.read_csv('tune/'+self.task_name+'/tune_val_'+note, index_col = 0)
            panel = pd.concat([panel, result], axis = 0, ignore_index = True)
        except:
            panel = result
        panel.to_csv('tune/'+self.task_name+'/tune_val_'+note)
        
        result = pd.DataFrame(index = [0], columns = title)
        result.iloc[0, :] = [self.factor_num, self.num_layers, self.dropout, self.r, self.beta,
                             self.lambda_param, self.lambda_F, self.lambda_D] + self.eval_test
        try:
            panel = pd.read_csv('tune/'+self.task_name+'/tune_test_'+note, index_col = 0)
            panel = pd.concat([panel, result], axis = 0, ignore_index = True)
        except:
            panel = result
        panel.to_csv('tune/'+self.task_name+'/tune_test_'+note)

    

# %% main
if __name__ == '__main__':
    torch.manual_seed(2023)
    exp = experiment(data_name = 'movielens', model_name = 'PMF', loss_name = 'A',
                     device = 'cuda',
                     factor_num = 16, num_layers = 4, dropout = 0.2,
                     r = 0.01, beta = 1,
                     lambda_param = 1e-3, lambda_F = 1e-2, lambda_D = 1e-2)
    exp.prepare_model()
    exp.fit()
    exp.evaluate()
    exp.record(note = 'regular')
    exp.prepare_model()
    exp.prepare_candidates()
    exp.prepare_influence()
    exp.prepare_selection()
    exp.fit()
    exp.evaluate()
    exp.record(note = 'influence')



