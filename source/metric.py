import recsys_metrics
import torch

# %% performance metrics
class rmse():
    def __init__(self):
        pass
    def __call__(self, model, interaction, 
                 user_gender = None, 
                 user_unknown = None,
                 user_unknown_mat = None,
                 user_unknown_cut = None,
                 item_tag = None):
        user, item, rating = \
            interaction[:, 0], \
            interaction[:, 1], \
            interaction[:, 2]
        pred = model(user.to('cuda'), item.to('cuda')).detach().to('cpu')
        return torch.sqrt(torch.mean((pred - rating)**2)).detach().item()

class precision():
    def __init__(self, k = 10):
        self.k = k
    def __call__(self, model, interaction, 
                 user_gender = None, 
                 user_unknown = None,
                 user_unknown_mat = None,
                 user_unknown_cut = None,
                 item_tag = None):
        
        # users = torch.unique(interaction[:, 0]).tolist()
        # k_list = torch.zeros((len(users)))
        # for i, user in enumerate(users):
        #     target = torch.zeros((len(user_unknown[user])))
        #     user_items = interaction[interaction[:, 0] == user, 1].tolist()
        #     for item in user_items:
        #         target[user_unknown[user].index(item)] = 1
        #     pred = model(torch.tensor([user]).repeat(len(user_unknown[user])), \
        #                  torch.tensor(user_unknown[user]))
        #     k_list[i] = recsys_metrics.precision(pred, target, k = self.k)
        # return k_list.mean().detach().item()
        
        user = torch.unique(interaction[:, 0])
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
        target_mat = torch.zeros((user.shape[0], total_eval))
        for i in range(user.shape[0]):
            target_mat[i, 0:user_unknown_cut[user[i]]] = 1
        return recsys_metrics.precision(pred_mat, target_mat, k = self.k).item()
    
class recall():
    def __init__(self, k = 10):
        self.k = k
    def __call__(self, model, interaction, 
                 user_gender = None, 
                 user_unknown = None,
                 user_unknown_mat = None,
                 user_unknown_cut = None,
                 item_tag = None):
        
        # users = torch.unique(interaction[:, 0]).tolist()
        # k_list = torch.zeros((len(users)))
        # for i, user in enumerate(users):
        #     target = torch.zeros((len(user_unknown[user])))
        #     user_items = interaction[interaction[:, 0] == user, 1].tolist()
        #     for item in user_items:
        #         target[user_unknown[user].index(item)] = 1
        #     pred = model(torch.tensor([user]).repeat(len(user_unknown[user])), \
        #                  torch.tensor(user_unknown[user]))
        #     k_list[i] = recsys_metrics.recall(pred, target, k = self.k)
        # return k_list.mean().detach().item()

        user = torch.unique(interaction[:, 0])
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
        target_mat = torch.zeros((user.shape[0], total_eval))
        for i in range(user.shape[0]):
            target_mat[i, 0:user_unknown_cut[user[i]]] = 1
        return recsys_metrics.recall(pred_mat, target_mat, k = self.k).item()

class f_score():
    def __init__(self, k = 10):
        self.k = k
    def __call__(self, model, interaction, 
                 user_gender = None, 
                 user_unknown = None,
                 user_unknown_mat = None,
                 user_unknown_cut = None,
                 item_tag = None):
        
        # users = torch.unique(interaction[:, 0]).tolist()
        # k_list = torch.zeros((len(users)))
        # for i, user in enumerate(users):
        #     target = torch.zeros((len(user_unknown[user])))
        #     user_items = interaction[interaction[:, 0] == user, 1].tolist()
        #     for item in user_items:
        #         target[user_unknown[user].index(item)] = 1
        #     pred = model(torch.tensor([user]).repeat(len(user_unknown[user])), \
        #                  torch.tensor(user_unknown[user]))
        #     precision_value, recall_value = \
        #         recsys_metrics.precision(pred, target, k = self.k), \
        #         recsys_metrics.recall(pred, target, k = self.k)
        #     k_list[i] = 2/(1/precision_value + 1/recall_value)
        # return k_list.mean().detach().item()
        
        user = torch.unique(interaction[:, 0])
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
        target_mat = torch.zeros((user.shape[0], total_eval))
        for i in range(user.shape[0]):
            target_mat[i, 0:user_unknown_cut[user[i]]] = 1
        precision_value, recall_value = \
            recsys_metrics.precision(pred_mat, target_mat, k = self.k).item(), \
            recsys_metrics.recall(pred_mat, target_mat, k = self.k).item()
        return 2/(1/(precision_value+1e-6) + 1/(recall_value+1e-6))
    
class ndcg():
    def __init__(self, k = 10):
        self.k = k
    def __call__(self, model, interaction, 
                 user_gender = None, 
                 user_unknown = None,
                 user_unknown_mat = None,
                 user_unknown_cut = None,
                 item_tag = None):
        
        # users = torch.unique(interaction[:, 0]).tolist()
        # k_list = torch.zeros((len(users)))
        # for i, user in enumerate(users):
        #     target = torch.zeros((len(user_unknown[user])))
        #     user_items = interaction[interaction[:, 0] == user, 1].tolist()
        #     for item in user_items:
        #         target[user_unknown[user].index(item)] = 1
        #     pred = model(torch.tensor([user]).repeat(len(user_unknown[user])), \
        #                  torch.tensor(user_unknown[user]))
        #     k_list[i] = recsys_metrics.normalized_dcg(pred, target, k = self.k)
        # return k_list.mean().detach().item()
    
        user = torch.unique(interaction[:, 0])
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
        target_mat = torch.zeros((user.shape[0], total_eval))
        for i in range(user.shape[0]):
            target_mat[i, 0:user_unknown_cut[user[i]]] = 1
        return recsys_metrics.normalized_dcg(pred_mat, target_mat, k = self.k).item()
        
# %% fairness metrics
class drmse():
    def __init__(self):
        pass
    def __call__(self, model, interaction, 
                 user_gender = None, 
                 user_unknown = None,
                 user_unknown_mat = None,
                 user_unknown_cut = None,
                 item_tag = None):
        metric = rmse()
        interaction_gender = user_gender[interaction[:, 0], 1]
        metric_0, metric_1 = \
            metric(model, interaction[interaction_gender == 0, :]), \
            metric(model, interaction[interaction_gender == 1, :])
        return abs(metric_0 - metric_1)

class dprecision():
    def __init__(self, k = 10):
        self.k = k
    def __call__(self, model, interaction, 
                 user_gender = None, 
                 user_unknown = None,
                 user_unknown_mat = None,
                 user_unknown_cut = None,
                 item_tag = None):
        metric = precision(k = self.k)
        interaction_gender = user_gender[interaction[:, 0], 1]
        metric_0, metric_1 = \
            metric(model, interaction[interaction_gender == 0, :],
                   user_unknown_mat = user_unknown_mat,
                   user_unknown_cut = user_unknown_cut), \
            metric(model, interaction[interaction_gender == 1, :],
                   user_unknown_mat = user_unknown_mat,
                   user_unknown_cut = user_unknown_cut)
        return abs(metric_0 - metric_1)
    
class drecall():
    def __init__(self, k = 10):
        self.k = k
    def __call__(self, model, interaction, 
                 user_gender = None, 
                 user_unknown = None,
                 user_unknown_mat = None,
                 user_unknown_cut = None,
                 item_tag = None):
        metric = recall(k = self.k)
        interaction_gender = user_gender[interaction[:, 0], 1]
        metric_0, metric_1 = \
            metric(model, interaction[interaction_gender == 0, :],
                   user_unknown_mat = user_unknown_mat,
                   user_unknown_cut = user_unknown_cut), \
            metric(model, interaction[interaction_gender == 1, :], 
                   user_unknown_mat = user_unknown_mat,
                   user_unknown_cut = user_unknown_cut)
        return abs(metric_0 - metric_1)
    
class df_score():
    def __init__(self, k = 10):
        self.k = k
    def __call__(self, model, interaction, 
                 user_gender = None, 
                 user_unknown = None,
                 user_unknown_mat = None,
                 user_unknown_cut = None,
                 item_tag = None):
        metric = f_score(k = self.k)
        interaction_gender = user_gender[interaction[:, 0], 1]
        metric_0, metric_1 = \
            metric(model, interaction[interaction_gender == 0, :], 
                   user_unknown_mat = user_unknown_mat,
                   user_unknown_cut = user_unknown_cut), \
            metric(model, interaction[interaction_gender == 1, :], 
                   user_unknown_mat = user_unknown_mat,
                   user_unknown_cut = user_unknown_cut)
        return abs(metric_0 - metric_1)
    
class dndcg():
    def __init__(self, k = 10):
        self.k = k
    def __call__(self, model, interaction, 
                 user_gender = None, 
                 user_unknown = None,
                 user_unknown_mat = None,
                 user_unknown_cut = None,
                 item_tag = None):
        metric = ndcg(k = self.k)
        interaction_gender = user_gender[interaction[:, 0], 1]
        metric_0, metric_1 = \
            metric(model, interaction[interaction_gender == 0, :], 
                   user_unknown_mat = user_unknown_mat,
                   user_unknown_cut = user_unknown_cut), \
            metric(model, interaction[interaction_gender == 1, :], 
                   user_unknown_mat = user_unknown_mat,
                   user_unknown_cut = user_unknown_cut)
        return abs(metric_0 - metric_1)

# %% diversity metrics (need modification!)
class topic_cover():
    def __init__(self, k = 10):
        self.k = k
    def __call__(self, model, interaction, 
                 user_gender = None, 
                 user_unknown = None,
                 user_unknown_mat = None,
                 user_unknown_cut = None,
                 item_tag = None):
        
        user_num = user_unknown_mat.shape[0]
        total_eval = user_unknown_mat.shape[1]
        k_list = torch.zeros((user_num))
        for user in range(user_num):
            pred = model(torch.tensor([user]).repeat(total_eval).to('cuda'), \
                         user_unknown_mat[user, :].to('cuda')).detach().to('cpu')
            topk_item = (user_unknown_mat[user, :][torch.topk(pred, self.k)[1]]).tolist()
            cover = set()
            for j, item in enumerate(topk_item):
                cover = cover.union(set(item_tag[item]))
            k_list[user] = len(cover)
        return k_list.mean().detach().item()
        
        # users = torch.unique(interaction[:, 0]).tolist()
        # k_list = torch.zeros((len(users)))
        # for i, user in enumerate(users):
        #     pred = model(torch.tensor([user]).repeat(len(user_unknown[user])), \
        #                  torch.tensor(user_unknown[user]))
        #     topk_item = (torch.tensor(user_unknown[user])[torch.topk(pred, self.k)[1]]).tolist()
        #     cover = set()
        #     for j, item in enumerate(topk_item):
        #         cover = cover.union(set(item_tag[item]))
        #     k_list[i] = len(cover)
        # return k_list.mean().detach().item()

# %% misc
class grmse():
    def __init__(self):
        pass
    def __call__(self, model, interaction, 
                 user_gender = None, 
                 user_unknown = None,
                 user_unknown_mat = None,
                 user_unknown_cut = None,
                 item_tag = None):
        metric = rmse()
        interaction_gender = user_gender[interaction[:, 0], 1]
        metric_0, metric_1 = \
            metric(model, interaction[interaction_gender == 0, :]), \
            metric(model, interaction[interaction_gender == 1, :])
        return metric_0, metric_1

class gprecision():
    def __init__(self, k = 10):
        self.k = k
    def __call__(self, model, interaction, 
                 user_gender = None, 
                 user_unknown = None,
                 user_unknown_mat = None,
                 user_unknown_cut = None,
                 item_tag = None):
        metric = precision(k = self.k)
        interaction_gender = user_gender[interaction[:, 0], 1]
        metric_0, metric_1 = \
            metric(model, interaction[interaction_gender == 0, :],
                   user_unknown_mat = user_unknown_mat,
                   user_unknown_cut = user_unknown_cut), \
            metric(model, interaction[interaction_gender == 1, :],
                   user_unknown_mat = user_unknown_mat,
                   user_unknown_cut = user_unknown_cut)
        return metric_0, metric_1
    
class grecall():
    def __init__(self, k = 10):
        self.k = k
    def __call__(self, model, interaction, 
                 user_gender = None, 
                 user_unknown = None,
                 user_unknown_mat = None,
                 user_unknown_cut = None,
                 item_tag = None):
        metric = recall(k = self.k)
        interaction_gender = user_gender[interaction[:, 0], 1]
        metric_0, metric_1 = \
            metric(model, interaction[interaction_gender == 0, :],
                   user_unknown_mat = user_unknown_mat,
                   user_unknown_cut = user_unknown_cut), \
            metric(model, interaction[interaction_gender == 1, :], 
                   user_unknown_mat = user_unknown_mat,
                   user_unknown_cut = user_unknown_cut)
        return metric_0, metric_1
    
class gf_score():
    def __init__(self, k = 10):
        self.k = k
    def __call__(self, model, interaction, 
                 user_gender = None, 
                 user_unknown = None,
                 user_unknown_mat = None,
                 user_unknown_cut = None,
                 item_tag = None):
        metric = f_score(k = self.k)
        interaction_gender = user_gender[interaction[:, 0], 1]
        metric_0, metric_1 = \
            metric(model, interaction[interaction_gender == 0, :], 
                   user_unknown_mat = user_unknown_mat,
                   user_unknown_cut = user_unknown_cut), \
            metric(model, interaction[interaction_gender == 1, :], 
                   user_unknown_mat = user_unknown_mat,
                   user_unknown_cut = user_unknown_cut)
        return metric_0, metric_1
    
class gndcg():
    def __init__(self, k = 10):
        self.k = k
    def __call__(self, model, interaction, 
                 user_gender = None, 
                 user_unknown = None,
                 user_unknown_mat = None,
                 user_unknown_cut = None,
                 item_tag = None):
        metric = ndcg(k = self.k)
        interaction_gender = user_gender[interaction[:, 0], 1]
        metric_0, metric_1 = \
            metric(model, interaction[interaction_gender == 0, :], 
                   user_unknown_mat = user_unknown_mat,
                   user_unknown_cut = user_unknown_cut), \
            metric(model, interaction[interaction_gender == 1, :], 
                   user_unknown_mat = user_unknown_mat,
                   user_unknown_cut = user_unknown_cut)
        return metric_0, metric_1





