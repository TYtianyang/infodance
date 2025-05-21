import torch
import tqdm
import os

# %% trainer: training the model
class trainer():
    
    # init
    def __init__(self, 
                 style = 'explicit',
                 loss_name = 'A', 
                 lambda_param = 1e-3,
                 lambda_F = 1e-2, 
                 lambda_D = 1e-2,
                 if_beutel = False):
        self.style = style
        self.loss_name = loss_name
        self.lambda_param = lambda_param
        self.lambda_F = lambda_F
        self.lambda_D = lambda_D
        self.if_beutel = if_beutel
    
    # fit
    def fit(self, loader, config = {'lr': 1e-0,
                                    'epochs': 1000,
                                    'batch_size': 32768,
                                    'gamma': 0.5,
                                    'patience': 50,
                                    'verbose': True}):

        optimizer = torch.optim.Adam(loader.model.parameters(),
                                     lr = config['lr'], 
                                     weight_decay = 0) # weight_decay will be introduced manually
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer = optimizer, 
                                                           gamma = config['gamma'])
            
        loss_min = float('inf')
        hesitate = 0
        iterator = tqdm.tqdm(range(config['epochs']), unit='epoch')

        for epoch in iterator:
        
            loader.generate(batch_size = config['batch_size'])
            
            user, pos_item, neg_item, rating = \
                loader.batch['user'], \
                loader.batch['pos_item'], \
                loader.batch['neg_item'], \
                loader.batch['rating']
            
            loss = loader.performance_proxy(user, pos_item, neg_item, rating)
            loss += self.lambda_param * loader.param_penalty()
            if 'F' in self.loss_name:
                loss_F = loader.fairness_proxy(user, pos_item, neg_item, rating)
                loss += self.lambda_F * loss_F
                
            if 'D' in self.loss_name:
                loss_D = loader.diversity_proxy(user, pos_item, neg_item, rating)
                loss += self.lambda_D * loss_D
                
            if self.if_beutel:
                loader.beutel_generate(batch_size = config['batch_size'])
                loss_corr = loader.beutel_proxy(loader.beutel_batch['item'],
                                                loader.beutel_batch['pos_user'],
                                                loader.beutel_batch['neg_user'], 
                                                loader.beutel_batch['rating'])
                loss += loss_corr
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if loss.item() > loss_min:
                hesitate += 1
            else:
                loss_min = loss.item()
                
            if hesitate >= config['patience']:
                scheduler.step()
                hesitate = 0
            
            if config['verbose']:
                iterator.set_description(
                    'Epoch {} | mean loss: {:.5f} lr: {:.5f}'.format(
                        epoch + 1, loss.item(), scheduler.get_last_lr()[0]))




