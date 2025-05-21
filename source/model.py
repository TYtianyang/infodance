import torch
import torch.nn as nn

# %% PMF model
class PMF(nn.Module):
    def __init__(self, user_num, item_num, 
                 style = 'explicit',
                 factor_num = 128):
        super(PMF, self).__init__()
        self.style = style
        self.user_num, self.item_num = user_num, item_num
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        
        self.last = torch.nn.Sigmoid()
        
    def forward(self, user, item):
        embed_user = self.embed_user(user)
        embed_item = self.embed_item(item)
        prediction = (embed_user * embed_item).sum(dim=-1)
        if self.style == 'explicit':
            return prediction
        elif self.style == 'implicit':
            return self.last(prediction)

# %% NCF model
class NCF(nn.Module):
    def __init__(self, user_num, item_num,
                 style = 'explicit',
                 factor_num = 128,
                 num_layers = 4,
                 dropout = 0.2):
        super(NCF, self).__init__()
        
        self.style = style
        self.user_num, self.item_num = user_num, item_num
        
        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.embed_user_MLP = nn.Embedding(
            user_num, factor_num * (2 ** (num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(
            item_num, factor_num * (2 ** (num_layers - 1)))
        
        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=dropout))
            MLP_modules.append(nn.Linear(input_size, input_size//2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)
        
        predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)
        self._init_weight_()
        
        self.last = torch.nn.Sigmoid()

    def _init_weight_(self):
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
        
        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(self.predict_layer.weight,
                                     a=1, nonlinearity='sigmoid')
            
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
                
    def forward(self, user, item):
        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)
        output_GMF = embed_user_GMF * embed_item_GMF
        
        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        output_MLP = self.MLP_layers(interaction)
        
        concat = torch.cat((output_GMF, output_MLP), -1)
        
        prediction = self.predict_layer(concat)
        if self.style == 'explicit':
            return prediction.view(-1)
        elif self.style == 'implicit':
            return self.last(prediction).view(-1)
        



