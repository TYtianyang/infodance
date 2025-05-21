import torch
import pickle
import os
import tqdm
import time
from functorch import make_functional_with_buffers, vmap, grad

os.chdir('/home/u9/tianyangxie/Documents/cf')
from source.model import PMF, NCF

# %% influence: influence function module
class influencer():
    
    # __init__
    def __init__(self, 
                 style = 'explicit',
                 loss_name = 'A', 
                 lambda_param = 1e-3,
                 lambda_F = 1e-2, 
                 lambda_D = 1e-2):
        self.style = style
        self.loss_name = loss_name
        self.lambda_param = lambda_param
        self.lambda_F = lambda_F
        self.lambda_D = lambda_D
    
    # first_component
    def first_component(self, loader, config = {'batch_size': 32768,
                                                'epochs': 500}):
        total_params = sum(p.numel() for p in loader.model.parameters())
        self.evaluation_grads = torch.zeros((total_params, 3))
        for i in tqdm.tqdm(range(config['epochs'])):
            loader.generate(batch_size = config['batch_size'])
            
            user, pos_item, neg_item, rating = \
                loader.batch['user'], \
                loader.batch['pos_item'], \
                loader.batch['neg_item'], \
                loader.batch['rating']
                
            loss_A = loader.performance_proxy(user, pos_item, neg_item, rating)
            evaluation_grad = list(torch.autograd.grad(loss_A, 
                                                       loader.model.parameters(), 
                                                       retain_graph = False))
            evaluation_grad = self.tflatten(evaluation_grad)
            self.evaluation_grads[:, 0] += evaluation_grad.to('cpu').squeeze()/config['epochs']
            
            loss_F = loader.fairness_proxy(user, pos_item, neg_item, rating)
            evaluation_grad = list(torch.autograd.grad(loss_F, 
                                                       loader.model.parameters(), 
                                                       retain_graph = False))
            evaluation_grad = self.tflatten(evaluation_grad)
            self.evaluation_grads[:, 1] += evaluation_grad.to('cpu').squeeze()/config['epochs']
            
            loss_D = loader.diversity_proxy(user, pos_item, neg_item, rating)
            evaluation_grad = list(torch.autograd.grad(loss_D, 
                                                       loader.model.parameters(), 
                                                       retain_graph = False))
            evaluation_grad = self.tflatten(evaluation_grad)
            self.evaluation_grads[:, 2] += evaluation_grad.to('cpu').squeeze()/config['epochs']
            
    # second component
    def second_component(self, loader, config = {'batch_size': 32768,
                                                 'lr': 0.1,
                                                 'epochs': 5000,
                                                 'warmup': True}):
        if config['warmup']:
            x_A, x_F, x_D = \
                self.vhps[:, 0:1].clone(), \
                self.vhps[:, 1:2].clone(), \
                self.vhps[:, 2:3].clone()
        else:
            x_A, x_F, x_D = \
                self.evaluation_grads[:, 0:1].clone(), \
                self.evaluation_grads[:, 1:2].clone(), \
                self.evaluation_grads[:, 2:3].clone()
        v_A, v_F, v_D = \
            self.evaluation_grads[:, 0:1].clone(), \
            self.evaluation_grads[:, 1:2].clone(), \
            self.evaluation_grads[:, 2:3].clone()
        for i in tqdm.tqdm(range(config['epochs'])):
            loader.generate(batch_size = config['batch_size'])
            
            user, pos_item, neg_item, rating = \
                loader.batch['user'], \
                loader.batch['pos_item'], \
                loader.batch['neg_item'], \
                loader.batch['rating']
                
            loss = loader.performance_proxy(user, pos_item, neg_item, rating)
            evaluation_grad = torch.autograd.grad(loss, 
                                                  loader.model.parameters(), 
                                                  create_graph = True)
            evaluation_grad = self.tflatten(evaluation_grad)
            Ax = torch.autograd.grad(evaluation_grad, 
                                     loader.model.parameters(),
                                     grad_outputs = x_A.to(loader.device))
            Ax = self.tflatten(Ax)
            x_A = v_A + x_A - config['lr']*Ax.to('cpu')
            
            loss = loader.performance_proxy(user, pos_item, neg_item, rating)
            evaluation_grad = torch.autograd.grad(loss, 
                                                  loader.model.parameters(), 
                                                  create_graph = True)
            evaluation_grad = self.tflatten(evaluation_grad)
            Ax = torch.autograd.grad(evaluation_grad, 
                                     loader.model.parameters(),
                                     grad_outputs = x_F.to(loader.device))
            Ax = self.tflatten(Ax)
            x_F = v_F + x_F - config['lr']*Ax.to('cpu')
            
            loss = loader.performance_proxy(user, pos_item, neg_item, rating)
            evaluation_grad = torch.autograd.grad(loss, 
                                                  loader.model.parameters(), 
                                                  create_graph = True)
            evaluation_grad = self.tflatten(evaluation_grad)
            Ax = torch.autograd.grad(evaluation_grad, 
                                     loader.model.parameters(),
                                     grad_outputs = x_D.to(loader.device))
            Ax = self.tflatten(Ax)
            x_D = v_D + x_D - config['lr']*Ax.to('cpu')

        self.vhps = torch.cat((x_A, x_F, x_D), dim = 1)
            
    # third component: compute per-sample influence
    def third_component(self, model, interaction, config = {'batch_size': 1248}):
        batch_num = int(interaction.shape[0]/config['batch_size']) + 1
        per_sample_gradient = torch.zeros((interaction.shape[0], 3))
        for i in tqdm.tqdm(range(batch_num)):
            per_sample_gradient_sub = self.per_sample_gradient(model, \
                interaction[(i*config['batch_size']):min((i+1)*config['batch_size'], interaction.shape[0]), :])
            per_sample_gradient_sub = self.tbflatten(per_sample_gradient_sub)
            per_sample_gradient_sub = torch.matmul(per_sample_gradient_sub, self.vhps)
            per_sample_gradient[(i*config['batch_size']):min((i+1)*config['batch_size'],\
                interaction.shape[0]), :] = per_sample_gradient_sub
        self.influence_value = - per_sample_gradient
        
    # tuple flatten: list to p x 1 tensor
    def tflatten(self, t):
        output = t[0].reshape((-1))
        for i in range(1, len(t)):
            output = torch.cat((output, t[i].reshape((-1))), 0)
        return output.reshape((-1 , 1))
    
    # tuple flatten but remain first dimension: list to n x p tensor
    def tbflatten(self, t):
        n = t[0].shape[0]
        output = t[0].reshape((n, -1))
        for i in range(1, len(t)):
            output = torch.cat((output, t[i].reshape((n, -1))), 1)
        return output
            
    # per sample gradient: third copmonent helper
    def per_sample_gradient(self, model, interaction):
        fmodel, params, buffers = make_functional_with_buffers(model.to('cpu'),
                                                               disable_autograd_tracking=True)
        if self.style == 'explicit':
            def compute_loss_stateless_model(params, buffers, interaction):
                interaction = interaction.reshape((-1, 3))
                user, item, rating = \
                    interaction[:, 0], \
                    interaction[:, 1], \
                    interaction[:, 2]
                pred = fmodel(params, buffers, user, item)
                loss = torch.sum((pred - rating)**2)
                return loss
        elif self.style == 'implicit':
            def compute_loss_stateless_model(params, buffers, interaction):
                interaction = interaction.reshape((-1, 2))
                user, item = \
                    interaction[:, 0], \
                    interaction[:, 1]
                pred = fmodel(params, buffers, user, item)
                loss = - torch.log(pred + 1e-8).sum()
                return loss
        ft_compute_grad = grad(compute_loss_stateless_model)
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0), randomness = 'different')
        ft_per_sample_grads = ft_compute_sample_grad(params, buffers, interaction)
        return ft_per_sample_grads
    
    # check vhps
    def check_vhps(self, loader, config = {'batch_size': 32768,
                                           'epochs': 500}):
        x_A, x_F, x_D = \
            self.vhps[:, 0:1], \
            self.vhps[:, 1:2], \
            self.vhps[:, 2:3]
        Ax_lhs = torch.zeros(self.evaluation_grads.shape)
        for i in tqdm.tqdm(range(config['epochs'])):
            loader.generate(batch_size = config['batch_size'])
            
            user, pos_item, neg_item, rating = \
                loader.batch['user'], \
                loader.batch['pos_item'], \
                loader.batch['neg_item'], \
                loader.batch['rating']
                
            loss = loader.performance_proxy(user, pos_item, neg_item, rating)
            evaluation_grad = torch.autograd.grad(loss, 
                                                  loader.model.parameters(), 
                                                  create_graph = True)
            evaluation_grad = self.tflatten(evaluation_grad)
            Ax = torch.autograd.grad(evaluation_grad, 
                                     loader.model.parameters(),
                                     grad_outputs = x_A.to(loader.device))
            Ax = self.tflatten(Ax)
            Ax_lhs[:, 0:1] += Ax.to('cpu')/config['epochs']
            
            loss = loader.performance_proxy(user, pos_item, neg_item, rating)
            evaluation_grad = torch.autograd.grad(loss, 
                                                  loader.model.parameters(), 
                                                  create_graph = True)
            evaluation_grad = self.tflatten(evaluation_grad)
            Ax = torch.autograd.grad(evaluation_grad, 
                                     loader.model.parameters(),
                                     grad_outputs = x_F.to(loader.device))
            Ax = self.tflatten(Ax)
            Ax_lhs[:, 1:2] += Ax.to('cpu')/config['epochs']
            
            loss = loader.performance_proxy(user, pos_item, neg_item, rating)
            evaluation_grad = torch.autograd.grad(loss, 
                                                  loader.model.parameters(), 
                                                  create_graph = True)
            evaluation_grad = self.tflatten(evaluation_grad)
            Ax = torch.autograd.grad(evaluation_grad, 
                                     loader.model.parameters(),
                                     grad_outputs = x_D.to(loader.device))
            Ax = self.tflatten(Ax)
            Ax_lhs[:, 2:3] += Ax.to('cpu')/config['epochs']

        print('The norm of distance is '+str(
            torch.linalg.norm(Ax_lhs - self.evaluation_grads, 'fro').item()))




