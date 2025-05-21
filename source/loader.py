import torch
import tqdm
import os

# %% loader 
class loader():
    
    # init
    def __init__(self, model, interaction, user_gender, item_tag, user_num, item_num, 
                 style = 'explicit', device = 'cuda', contrast = 'bpr'):
        self.device = device
        self.model = model.to(device)
        self.interaction = interaction
        self.user_gender = user_gender.to(self.device)
        self.item_tag = item_tag
        self.construct_item_tag_mat()
        self.user_num, self.item_num = user_num, item_num
        self.style = style
        
        self.loss_mse = torch.nn.MSELoss()
        if contrast == 'rank':
            self.loss_rank = torch.nn.MarginRankingLoss(margin = 1)
        elif contrast == 'bpr':
            def contrast_func(pos_pred, neg_pred, target):
                return -torch.log(torch.sigmoid(target * (pos_pred - neg_pred))).mean()
            self.loss_rank = contrast_func
                
            
    # construct_item_tag_mat
    def construct_item_tag_mat(self):
        item_tag = self.item_tag
        tag_union = set()
        for i in range(len(item_tag)):
            tag_union = tag_union.union(item_tag[i])
        tag_union = list(tag_union)
        mat = torch.zeros((len(item_tag), len(tag_union)))
        for i in range(len(item_tag)):
            for j in range(len(item_tag[i])):
                mat[i, tag_union.index(item_tag[i][j])] = 1
        self.item_tag_mat = mat.to(self.device)
    
    # generate
    def generate(self, batch_size = 32768):
        ind = torch.randint(low = 0, 
                            high = self.interaction.shape[0], 
                            size = (batch_size,))
        pos_interaction = self.interaction[ind, :]
        user, pos_item = \
            pos_interaction[:, 0].to(self.device), \
            pos_interaction[:, 1].to(self.device)
        if self.style == 'explicit':
            rating = pos_interaction[:, 2].float().to(self.device)
        else:
            rating = None
            
        neg_item = torch.randint(low = 0,
                                 high = self.item_num,
                                 size = (batch_size,),
                                 device = self.device)
        
        gender = self.user_gender[user.to('cpu')]
        
        if torch.unique(gender).shape[0] < 2:
            self.genderate()
        else:
            self.batch = {'user': user,
                          'pos_item': pos_item,
                          'neg_item': neg_item,
                          'rating': rating}
            
    # beutel_generate (baseline beutel)
    def beutel_generate(self, batch_size = 32768):
        ind = torch.randint(low = 0, 
                            high = self.interaction.shape[0], 
                            size = (batch_size,))
        pos_interaction = self.interaction[ind, :]
        pos_user, item = \
            pos_interaction[:, 0].to(self.device), \
            pos_interaction[:, 1].to(self.device)
        if self.style == 'explicit':
            rating = pos_interaction[:, 2].float().to(self.device)
        else:
            rating = None
            
        neg_user = torch.randint(low = 0,
                                 high = self.user_num,
                                 size = (batch_size,),
                                 device = self.device)
        
        self.beutel_batch = {'pos_user': pos_user,
                             'neg_user': neg_user,
                             'item': item,
                             'rating': rating}
        
    # param penalty
    def param_penalty(self):
        penalty = 0
        for p in self.model.parameters():
            penalty += (p**2).sum()
        return torch.sqrt(penalty)
        
    # performance_proxy
    def performance_proxy(self, user, pos_item, neg_item, rating = None):
        if self.style == 'explicit':
            pred = self.model(user, pos_item)
            return self.loss_mse(pred, rating)
        elif self.style == 'implicit':
            pos_pred, neg_pred = \
                self.model(user, pos_item), self.model(user, neg_item)
            return self.loss_rank(pos_pred, neg_pred, target = torch.ones(pos_pred.shape, device = self.device))
        
    # fairness_proxy
    def fairness_proxy(self, user, pos_item, neg_item, rating):
        gender = self.user_gender[user]
        gender_ind0, gender_ind1 = \
            ((gender == 0).nonzero(as_tuple=True)[0]), \
            ((gender == 1).nonzero(as_tuple=True)[0])
        if self.style == 'explicit':
            loss0, loss1 = \
                self.performance_proxy(user[gender_ind0], 
                                       pos_item[gender_ind0], 
                                       neg_item[gender_ind0], 
                                       rating[gender_ind0]), \
                self.performance_proxy(user[gender_ind1], 
                                       pos_item[gender_ind1], 
                                       neg_item[gender_ind1], 
                                       rating[gender_ind1])
        elif self.style == 'implicit':
            loss0, loss1 = \
                self.performance_proxy(user[gender_ind0], 
                                       pos_item[gender_ind0], 
                                       neg_item[gender_ind0], 
                                       None), \
                self.performance_proxy(user[gender_ind1], 
                                       pos_item[gender_ind1], 
                                       neg_item[gender_ind1], 
                                       None)
        return torch.abs(loss0 - loss1)

    # diversity_proxy
    def diversity_proxy(self, user, pos_item, neg_item, rating):
        pos_pred, neg_pred = \
            self.model(user, pos_item), self.model(user, neg_item)
        target = torch.where(self.item_tag_mat[pos_item, :].sum(dim=1) >= 
                             self.item_tag_mat[neg_item, :].sum(dim=1), 1, -1)
        return self.loss_rank(pos_pred, neg_pred, target)
        
    # beutel_proxy (baseline beutel)
    def beutel_proxy(self, item, pos_user, neg_user, rating = None):
        pos_gender, neg_gender = self.user_gender[pos_user, 1], self.user_gender[neg_user, 1]
        pos_pred, neg_pred = self.model(pos_user, item), self.model(neg_user, item)
        pos_rating = torch.ones(item.shape, device = self.device)
        neg_rating = torch.zeros(pos_rating.shape, device = self.device)
            
        A = ((pos_pred - neg_pred) * (pos_rating - neg_rating)).reshape((1, -1))
        B = ((pos_gender - neg_gender) * (pos_rating - neg_rating)).reshape((1, -1))
        corr = torch.abs(torch.corrcoef(torch.cat((A, B), dim = 0))[0, 1])
        return corr/item.shape[0]
    
        
        
        
        
        
