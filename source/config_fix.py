class config_fix():
    
    def __init__(self, task_name = 'movielens_PMF_A'):
        
        if 'movielens_PMF' in task_name:
            
            self.config_model = {'factor_num': 16,
                                 'num_layers': 4,
                                 'dropout': 0.2}
            
            self.config_trainer_static = {'lr': 1e-2,
                                          'epochs': 500,
                                          'gamma': 0.5,
                                          'batch_num': 1,
                                          'patience': 20, 
                                          'verbose': True}
            
            self.config_trainer_dynamic = {'lr': 1e-3,
                                           'epochs': 5,
                                           'gamma': 0.5,
                                           'batch_num': 1,
                                           'patience': 20, 
                                           'verbose': False}
            
            self.config_influence_static = {'method': 'SGD',
                                            'rtol': 1e-3,
                                            'atol': 1e-6,
                                            'maxiter': 1000,
                                            'verbose': True,
                                            'steps': 5000,
                                            'sample_size': 1000,
                                            'tol': 1e-6}
            
            self.config_influence_dynamic =  {'method': 'SGD',
                                              'rtol': 1e-3,
                                              'atol': 1e-6,
                                              'maxiter': 10,
                                              'verbose': False,
                                              'steps': 1000,
                                              'sample_size': 1000,
                                              'tol': 1e-6}
            
        elif 'movielens_NCF' in task_name:
            
            self.config_model = {'factor_num': 16,
                                 'num_layers': 4,
                                 'dropout': 0.2}
            
            self.config_trainer_static = {'lr': 1e-3,
                                          'epochs': 1000,
                                          'gamma': 0.5,
                                          'batch_num': 1,
                                          'patience': 20, 
                                          'verbose': True}
            
            self.config_trainer_dynamic = {'lr': 1e-4,
                                           'epochs': 5,
                                           'gamma': 0.5,
                                           'batch_num': 1,
                                           'patience': 20, 
                                           'verbose': False}
            
            self.config_influence_static = {'method': 'SGD',
                                            'rtol': 1e-3,
                                            'atol': 1e-6,
                                            'maxiter': 1000,
                                            'verbose': True,
                                            'steps': 500,
                                            'sample_size': 1000,
                                            'tol': 1e-6}
            
            self.config_influence_dynamic =  {'method': 'SGD',
                                              'rtol': 1e-3,
                                              'atol': 1e-6,
                                              'maxiter': 10,
                                              'verbose': False,
                                              'steps': 1000,
                                              'sample_size': 1000,
                                              'tol': 1e-6}
            
        elif 'lastfm_PMF' in task_name:
            
            self.config_model = {'factor_num': 128,
                                 'num_layers': 4,
                                 'dropout': 0.2}

            self.config_trainer_static = {'lr': 1e-2,
                                          'epochs': 500,
                                          'gamma': 0.5,
                                          'batch_num': 1,
                                          'patience': 20, 
                                          'verbose': True}
            
            self.config_trainer_dynamic = {'lr': 1e-3,
                                           'epochs': 5,
                                           'gamma': 0.5,
                                           'batch_num': 1,
                                           'patience': 20, 
                                           'verbose': False}
            
            self.config_influence_static = {'method': 'CG_torch_double',
                                            'rtol': 1e-3,
                                            'atol': 1e-5,
                                            'maxiter': 100,
                                            'verbose': True}
            
            self.config_influence_dynamic =  {'method': 'CG_torch_double',
                                              'rtol': 1e-3,
                                              'atol': 1e-5,
                                              'maxiter': 3,
                                              'verbose': False}
            
        elif 'lastfm_NCF' in task_name:
            
            self.config_model = {'factor_num': 128,
                                 'num_layers': 4,
                                 'dropout': 0.2}
            
            self.config_trainer_static = {'lr': 1e-3,
                                          'epochs': 500,
                                          'gamma': 0.5,
                                          'batch_num': 8,
                                          'patience': 20, 
                                          'verbose': True}
            
            self.config_trainer_dynamic = {'lr': 1e-4,
                                           'epochs': 5,
                                           'gamma': 0.5,
                                           'batch_num': 8,
                                           'patience': 20, 
                                           'verbose': False}
            
            self.config_influence_static = {'method': 'CG_torch_double',
                                            'rtol': 1e-3,
                                            'atol': 1e-5,
                                            'maxiter': 100,
                                            'verbose': True}
            
            self.config_influence_dynamic =  {'method': 'CG_torch_double',
                                              'rtol': 1e-3,
                                              'atol': 1e-5,
                                              'maxiter': 3,
                                              'verbose': False}
