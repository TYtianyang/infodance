import torch
import torch.nn as nn
import os
os.chdir('/home/u9/tianyangxie/Documents/cf')

# %% knowledge distillation class
class knowledge_distill(nn.Module):
    
    # init function
    def __init__(self, model,
                 evaluation_func_num = 3,
                 rating_num = 5,
                 style = 'explicit',
                 config = {'num_layers': 5,
                           'dropout': 0.2,
                           'rating_dim': 10,
                           'distill': False}):
        
        super(knowledge_distill, self).__init__()
        
        # register
        self.style = style
        self.distill = config['distill']
        self.evaluation_func_num = evaluation_func_num
        rating_dim = config['rating_dim']
        
        # copy some parameters from model
        if model.__class__.__name__ == 'PMF':
            user_num = model.embed_user.weight.data.shape[0]
            item_num = model.embed_item.weight.data.shape[0]
            factor_num = model.embed_user.weight.data.shape[1]
        if model.__class__.__name__ == 'NCF':
            user_num = model.embed_user_GMF.weight.data.shape[0]
            item_num = model.embed_item_GMF.weight.data.shape[0]
            factor_num = \
                model.embed_user_GMF.weight.data.shape[1] + \
                model.embed_user_MLP.weight.data.shape[1]
                
        # remember parameters
        self.user_num = user_num
        self.item_num = item_num
        self.rating_num = rating_num
            
        # setup projection embedding
        if model.__class__.__name__ == 'PMF':
            self.project_user = nn.Embedding(user_num, factor_num).requires_grad_(False)
            self.project_user.weight.data = model.embed_user.weight.data
            self.project_item = nn.Embedding(item_num, factor_num).requires_grad_(False)
            self.project_item.weight.data = model.embed_item.weight.data
        if model.__class__.__name__ == 'NCF':
            self.project_user = nn.Embedding(user_num, factor_num).requires_grad_(False)
            self.project_user.weight.data = torch.cat((model.embed_user_GMF.weight.data,
                                                       model.embed_user_MLP.weight.data), dim = 1)
            self.project_item = nn.Embedding(item_num, factor_num).requires_grad_(False)
            self.project_item.weight.data = torch.cat((model.embed_user_GMF.weight.data,
                                                       model.embed_user_MLP.weight.data), dim = 1)
            
        # setup network (distill v.s. not distill)
        if self.distill:
            if self.style == 'explicit':
                self.project_rating = nn.Embedding(rating_num + 1,
                                                   rating_dim,
                                                   padding_idx = 0)
                factor_total = factor_num + rating_dim
            else:
                factor_total = factor_num
                
            self.mlp_user = nn.ModuleList()
            for i in range(config['num_layers'] - 1):
                self.mlp_user.append(nn.Dropout(p = config['dropout']))
                self.mlp_user.append(nn.Linear(factor_total, factor_total))
                self.mlp_user.append(nn.ReLU())
            self.mlp_user.append(nn.Dropout(p = config['dropout']))
            self.mlp_user.append(nn.Linear(factor_total, evaluation_func_num))
            
            self.mlp_item = nn.ModuleList()
            for i in range(config['num_layers'] - 1):
                self.mlp_item.append(nn.Dropout(p = config['dropout']))
                self.mlp_item.append(nn.Linear(factor_total, factor_total))
                self.mlp_item.append(nn.ReLU())
            self.mlp_item.append(nn.Dropout(p = config['dropout']))
            self.mlp_item.append(nn.Linear(factor_total, evaluation_func_num))

        else:
            if self.style == 'explicit':
                self.project_rating = nn.Embedding(rating_num + 1,
                                                   rating_dim,
                                                   padding_idx = 0)
                factor_total = 2*factor_num + rating_dim
            else:
                factor_total = 2*factor_num
                
            self.mlp = nn.ModuleList()
            for i in range(config['num_layers'] - 1):
                self.mlp.append(nn.Dropout(p = config['dropout']))
                self.mlp.append(nn.Linear(factor_total, factor_total))
                self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Dropout(p = config['dropout']))
            self.mlp.append(nn.Linear(factor_total, evaluation_func_num))
            
    # forward_user
    def forward_user(self, interaction):
        user_mat = self.project_user(interaction[:, 0])
        
        if self.style == 'explicit':
            rating_mat = self.project_rating(interaction[:, 2])
            x = torch.cat((user_mat, rating_mat), dim = 1)
        else:
            x = user_mat
            
        for f in self.mlp_user:
            x = f(x)
        return x
    
    # forward_item
    def forward_item(self, interaction):
        item_mat = self.project_item(interaction[:, 1])
        
        if self.style == 'explicit':
            rating_mat = self.project_rating(interaction[:, 2])
            x = torch.cat((item_mat, rating_mat), dim = 1)
        else:
            x = item_mat
            
        for f in self.mlp_item:
            x = f(x)
        return x
    
    # forward_total
    def forward_total(self, interaction):
        user_mat = self.project_user(interaction[:, 0])
        item_mat = self.project_user(interaction[:, 1])
        
        if self.style == 'explicit':
            rating_mat = self.project_rating(interaction[:, 2])
            x = torch.cat((user_mat, item_mat, rating_mat), dim = 1)
        else:
            x = torch.cat((user_mat, item_mat), dim = 1)
            
        for f in self.mlp:
            x = f(x)
        return x
    
    # forward
    def forward(self, interaction):
        if self.distill:
            return self.forward_user(interaction) + self.forward_item(interaction)
        else:
            return self.forward_total(interaction)
    
    # create_interaction
    def create_interaction(self, size = 1e+5):
        user_sample = torch.randint(high = self.user_num,
                                    size = (int(size), 1))
        item_sample = torch.randint(high = self.item_num,
                                    size = (int(size), 1))
        if self.style == 'explicit':
            rating_sample = torch.randint(high = self.rating_num,
                                          size = (int(size), 1))
            return torch.cat((user_sample, item_sample, rating_sample), dim = 1)
        else:
            return torch.cat((user_sample, item_sample), dim = 1)
    
    # create_label
    def create_label(self, model, infl, interaction,
                     user_unknown = None,
                     batch_num = 1):
        return infl.third_component(model, interaction, 
                                    user_unknown = user_unknown,
                                    batch_num = batch_num)
    
    # fit
    def fit(self, model, infl,
            user_unknown = None,
            config = {'train_size': 1e+5,
                      'val_size': 1e+5,
                      'lr': 1e-2,
                      'weight_decay': 0,
                      'epochs': 100,
                      'gamma': 0.5,
                      'patience': 5,
                      'verbose': False}):

        optimizer = torch.optim.Adam(self.parameters(),
                                     lr = config['lr'], 
                                     weight_decay = config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer = optimizer, 
                                                           gamma = config['gamma'])

        interaction_train = self.create_interaction(config['train_size'])
        interaction_val = self.create_interaction(config['val_size'])
        label_train = self.create_label(model, infl, interaction_train,
                                        user_unknown = user_unknown)
        label_val = self.create_label(model, infl, interaction_val,
                                      user_unknown = user_unknown)
            
        acc_max = 0
        hesitate = 0
        
        for i in range(config['epochs']):
            pred_train = self.forward(interaction_train)
            pred_val = self.forward(interaction_val).detach()
            
            loss = nn.functional.binary_cross_entropy(torch.sigmoid(pred_train).reshape((-1)), 
                                                      (torch.sign(label_train) == 1).float().reshape((-1)))
            acc_train = torch.mean((torch.sign(label_train) == torch.sign(pred_train.detach())).float(), dim = 0)
            acc_val = torch.mean((torch.sign(label_val) == torch.sign(pred_val)).float(), dim = 0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            if acc_val.mean().item() < acc_max:
                hesitate += 1
            else:
                acc_max = acc_val.mean().item()
                
            if hesitate >= config['patience']:
                scheduler.step()
                hesitate = 0
            
            if config['verbose']:
                print(str(i) + ':Loss:  ' + str(loss.item()) \
                      + '  acc_train  ' + str([round(i, 4) for i in acc_train.tolist()]) \
                      + '  acc_val  ' + str([round(i, 4) for i in acc_val.tolist()]) \
                      + '  lr:  ' + str(scheduler.get_lr()))
    
    # recover
    def recover(self):
        user_vec = torch.arange(self.user_num)
        item_vec = torch.arange(self.item_num)
        rating_vec = torch.arange(self.rating_num)
        if self.distill:
            if self.style == 'explicit':
                interaction = torch.zeros((self.user_num * self.rating_num, 3))
                interaction[:, 0] = torch.cartesian_prod(user_vec, rating_vec)[:, 0]
                interaction[:, 2] = torch.cartesian_prod(user_vec, rating_vec)[:, 1]
                influence_user = self.forward_user(interaction).reshape((self.user_num,
                                                                          1,
                                                                          self.rating_num, 
                                                                          self.evaluation_func_num))
                interaction = torch.zeros((self.item_num * self.rating_num, 3))
                interaction[:, 1] = torch.cartesian_prod(item_vec, rating_vec)[:, 0]
                interaction[:, 2] = torch.cartesian_prod(item_vec, rating_vec)[:, 1]
                influence_item = self.forward_item(interaction).reshape((1,
                                                                          self.item_num,
                                                                          self.rating_num, 
                                                                          self.evaluation_func_num))
                influence = influence_user + influence_item
            else:
                interaction = torch.zeros((self.user_num, 3))
                interaction[:, 0] = user_vec
                influence_user = self.forward_user(interaction).reshape((self.user_num,
                                                                          1, 
                                                                          self.evaluation_func_num))
                interaction = torch.zeros((self.item_num, 3))
                interaction[:, 1] = item_vec
                influence_item = self.forward_item(interaction).reshape((1,
                                                                          self.item_num,
                                                                          self.evaluation_func_num))
                influence = influence_user + influence_item
        else:
            if self.style == 'explicit':
                interaction = torch.cartesian_prod(user_vec, item_vec, rating_vec)
                influence = self.forward_total(interaction).reshape((self.user_num,
                                                                     self.item_num,
                                                                     self.rating_num,
                                                                     self.evaluation_func_num))
            else:
                interaction = torch.cartesian_prod(user_vec, item_vec)
                influence = self.forward_total(interaction).reshape((self.user_num,
                                                                     self.item_num,
                                                                     self.evaluation_func_num))
        return influence
            
            
            
            
            
            
            
            
            
            
            
    