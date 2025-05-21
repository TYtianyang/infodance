import torch

# %% performance evaluation function
class performance_proxy():
    def __init__(self, style = 'explicit', neg_portion = 2, device = 'cpu'):
        self.style = style
        self.neg_portion = neg_portion
        self.device = device
    def __call__(self, model, interaction,
                 user_gender = None,
                 user_unknown = None,
                 item_tag = None):
        if self.style == 'explicit':
            user, item, rating = \
                interaction[:, 0].to(self.device), \
                interaction[:, 1].to(self.device), \
                interaction[:, 2].to(self.device)
            pred = model(user, item)
            loss = torch.mean((pred - rating)**2)
            return loss
        elif self.style == 'implicit':
            user, item = \
                interaction[:, 0].to(self.device), \
                interaction[:, 1].to(self.device)
            pred_positive = model(user, item)
            
            user_list = torch.unique(interaction[:, 0].to(self.device)).tolist()
            pred_negative = torch.tensor([], device = self.device)
            for i, user in enumerate(user_list):
                pred_negative = torch.cat((pred_negative,
                                           model(torch.tensor([user], device = self.device).repeat(len(user_unknown[user])), \
                                                        torch.tensor(user_unknown[user], device = self.device))))
            lamb = self.neg_portion * pred_positive.shape[0] / pred_negative.shape[0]
            self.pred_positive = pred_positive
            self.pred_negative = pred_negative
            loss = (- torch.log(pred_positive + 1e-8).sum() - lamb * torch.log(1 - pred_negative + 1e-8).sum())/pred_positive.shape[0]
            return loss
        
# %% fairness evaluation function
class fairness_proxy():
    def __init__(self, style = 'explicit', neg_portion = 2, device = 'cpu'):
        self.style = style
        self.neg_portion = neg_portion
        self.device = device
    def __call__(self, model, interaction,
                 user_gender = None,
                 user_unknown = None,
                 item_tag = None):
        performance_eval = performance_proxy(style = self.style, neg_portion = self.neg_portion, device = self.device)
        interaction_gender = user_gender[interaction[:, 0], 1]
        performance_0, performance_1 = \
            performance_eval(model, interaction[interaction_gender == 0, :], user_unknown = user_unknown), \
            performance_eval(model, interaction[interaction_gender == 1, :], user_unknown = user_unknown)
        return abs(performance_0 - performance_1)

# %% diversity evaluation function
# class diversity_proxy():
#     def __init__(self, style = 'explicit', kb = 20, sample_size = -1, device = 'cpu'):
#         self.style = style
#         self.kb = kb # fake, we don't use it
#         self.sample_size = sample_size
#         self.device = device
#         self.item_tag_mat = None
#         self.relu = torch.nn.ReLU()
#     def __call__(self, model, interaction,
#                   user_gender = None,
#                   user_unknown = None,
#                   item_tag = None):
#         sample_size = self.sample_size
        
#         if self.item_tag_mat == None:
#             union_tag = set()
#             for i in range(len(item_tag)):
#                 union_tag = union_tag.union(set(item_tag[i]))
#             self.item_tag_mat = torch.zeros((len(item_tag)), len(union_tag))
#             self.union_tag = list(union_tag)
        
#             for i in range(len(item_tag)):
#                 ind = [self.union_tag.index(tag) for tag in item_tag[i]]
#                 self.item_tag_mat[i, ind] = 1
                
#             self.item_tag_mat = self.item_tag_mat.to(self.device)
            
#         item = torch.arange(model.item_num, device = self.device)
            
#         if sample_size != -1:
#             user = torch.randperm(model.user_num, device = self.device)[0:sample_size]
#         else:
#             user = torch.arange(model.user_num, device = self.device)
        
#         interaction_cat = torch.cartesian_prod(user, item)
#         pred = model(interaction_cat[:, 0], interaction_cat[:, 1]).reshape((-1, model.item_num))
#         score = self.relu(torch.matmul(pred, self.item_tag_mat)).pow(0.1)
#         return - score.mean()

# class diversity_proxy():
#     def __init__(self, style = 'explicit', kb = 20):
#         self.style = style
#         self.kb = kb
#         self.softmax = torch.nn.Softmax(dim = 0)
#         self.item_tag_mat = None
#     def __call__(self, model, interaction,
#                   user_gender = None,
#                   user_unknown = None,
#                   item_tag = None):
#         if self.item_tag_mat == None:
#             union_tag = set()
#             for i in range(len(item_tag)):
#                 union_tag = union_tag.union(set(item_tag[i]))
#             self.item_tag_mat = torch.zeros((len(item_tag)), len(union_tag))
#             self.union_tag = list(union_tag)
        
#             for i in range(len(item_tag)):
#                 ind = [self.union_tag.index(tag) for tag in item_tag[i]]
#                 self.item_tag_mat[i, ind] = 1
                
#             rate = self.item_tag_mat.mean(dim = 0).reshape((1, -1))
#             self.item_tag_seq = (self.item_tag_mat/torch.sqrt(rate)).mean(dim = 1)
        
#         users = torch.unique(interaction[:, 0]).tolist()
#         k_list = torch.zeros((len(users)))
#         for i, user in enumerate(users):
#             item_list = torch.cat((interaction[interaction[:, 0] == user, 1], torch.tensor(user_unknown[user])))
#             pred = model(torch.tensor([user]).repeat(item_list.shape[0]), \
#                           item_list)
#             k_list[i] = (self.item_tag_seq[item_list] * pred).mean()
#             # topk_pred, topk_item_ind = torch.topk(pred, self.kb)
#             # topk_item = item_list[topk_item_ind]
#             # k_list[i] = (self.item_tag_seq[topk_item] * topk_pred).mean()
#         return - k_list.mean()

class diversity_proxy():
    def __init__(self, style = 'explicit', kb = 20, sample_size = -1, device = 'cpu'):
        self.style = style
        self.kb = kb # fake, we don't use it
        self.sample_size = sample_size # fake, we don't use it
        self.device = device
        self.item_tag_mat = None
    def __call__(self, model, interaction,
                  user_gender = None,
                  user_unknown = None,
                  item_tag = None):
        if self.item_tag_mat == None:
            union_tag = set()
            for i in range(len(item_tag)):
                union_tag = union_tag.union(set(item_tag[i]))
            self.item_tag_mat = torch.zeros((len(item_tag)), len(union_tag))
            self.union_tag = list(union_tag)
        
            for i in range(len(item_tag)):
                ind = [self.union_tag.index(tag) for tag in item_tag[i]]
                self.item_tag_mat[i, ind] = 1
                
            rate = self.item_tag_mat.sum(dim = 0).reshape((1, -1)).pow(2)
            self.item_tag_seq = (self.item_tag_mat/rate).sum(dim = 1).to(self.device)
            
        user, item = interaction[:, 0].to(self.device), interaction[:, 1].to(self.device)
        pred = model(user, item)
        return - (self.item_tag_seq[item] * pred).mean()
        
# class diversity_proxy():
#     def __init__(self, style = 'explicit', kb = 20, device = 'cpu'):
#         self.style = style
#         self.kb = kb
#         self.softmax = torch.nn.Softmax(dim = 0)
#         self.device = device
#     def __call__(self, model, interaction,
#                   user_gender = None,
#                   user_unknown = None,
#                   item_tag = None):
#         users = torch.unique(interaction[:, 0].to(self.device)).tolist()
#         k_list = torch.zeros((len(users)), device = self.device)
#         for i, user in enumerate(users):
#             pred = model(torch.tensor([user], device = self.device).repeat(len(user_unknown[user])), \
#                           torch.tensor(user_unknown[user], device = self.device))
#             pred = self.softmax(torch.topk(pred, self.kb)[0])
#             topk_item = (torch.tensor(user_unknown[user])[torch.topk(pred, self.kb)[1]]).tolist()
#             all_tag = set()
#             for j, item in enumerate(topk_item):
#                 all_tag = all_tag.union(set(item_tag[item]))
#             all_tag = list(all_tag)
#             all_score = torch.zeros((len(all_tag)))
#             for j, item in enumerate(topk_item):
#                 tag_ind = [all_tag.index(tag) for tag in item_tag[item]]
#                 all_score[tag_ind] = all_score[tag_ind] + pred[j]
#             all_score = torch.sqrt(torch.clamp(all_score, min = 1e-4, max = 1))
#             k_list[i] = all_score.sum()
#         return - k_list.mean()






