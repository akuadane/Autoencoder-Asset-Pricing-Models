import os
import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset



MAX_EPOCH = 200
class Autoencoder_Base(nn.Module):
    def __init(self,name,hidden_size, dropout=0.2, lr=0.001, omit_char=[], device='cuda'):
                # initial train, valid and test periods are default accroding to original paper
        nn.Module.__init__(self)

        self.train_period = [19570101, 19741231]
        self.valid_period = [19750101, 19861231]
        self.test_period  = [19870101, 19871231]

        self.name = name
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.encoder = None
        self.decoder = None
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.001)
        self.criterion = nn.MSELoss().to(device)
        self.omit_char = omit_char
        
        
        self.device = device

        self.datashare_chara = pd.read_pickle('./data/datashare_re.pkl').astype(np.float64)
        self.p_charas = pd.read_pickle('./data/p_charas.pkl').astype(np.float64).reset_index()
        self.portfolio_ret=  pd.read_pickle('./data/portfolio_ret.pkl').astype(np.float64)
        self.mon_ret = pd.read_pickle('./data/month_ret.pkl').astype(np.float64)

        self.train_dataloader = None
        self.valid_dataloader = None
        self.test_dataloader = None
    
    def forward(self,pfret):
        pass
    
    def _get_item(self, month):
        if month not in self.p_charas['DATE'].values:
            # find the closest month in p_charas to month
            month = self.p_charas['DATE'].values[np.argmin(np.abs(self.p_charas['DATE'].values - month))]
            
        beta_nn_input = self.p_charas.loc[self.p_charas['DATE'] == month][CHARAS_LIST] # (94, 94)
        labels = self.portfolio_ret.loc[self.portfolio_ret['DATE'] == month][CHARAS_LIST].T.values # (94, 1)
        beta_nn_input['ret-rf'] = labels
        align_df = beta_nn_input.copy(deep=False).dropna()
            
        factor_nn_input = self.portfolio_ret.loc[self.portfolio_ret['DATE'] == month][CHARAS_LIST]
         
        # exit(0) if there is any nan in align_df
        if align_df.isnull().values.any():
            assert False, f'There is nan in align_df of : {month}'
        # return stock index (L), beta_nn_input (94*94=P*N), factor_nn_input (94*1=P*1), labels (94, = N,)
        return align_df.index, align_df.values[:, :-1].T, factor_nn_input.T.values , align_df.values[:, -1].T
    
    def dataloader(self, period): 
        mon_list = pd.read_pickle('data/mon_list.pkl')
        mon_list = mon_list.loc[(mon_list >= period[0]) & (mon_list <= period[1])]
        # beta_nn_input_set = []
        factor_nn_input_set = []
        label_set = []
        for mon in mon_list:
            _, _beta_input, _factor_input, label =  self._get_item(mon)
            # beta_nn_input_set.append(_beta_input)
            factor_nn_input_set.append(_factor_input)
            label_set.append(label)
            
        # beta_nn_input_set = torch.tensor(beta_nn_input_set, dtype=torch.float32).to(self.device)
        factor_nn_input_set = torch.tensor(factor_nn_input_set, dtype=torch.float32).to(self.device)
        label_set = torch.tensor(label_set, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(factor_nn_input_set, label_set)   
        return DataLoader(dataset, batch_size=1, shuffle=True)
    

    # train_one_epoch
    def __train_one_epoch(self):
        epoch_loss = 0.0
        for i, (factor_nn_input, labels) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            # beta_nn_input reshape: (1, 94, 94) -> (94, 94) (1*P*N => N*P)
            # factor_nn_input reshape: (1, 94, 1) -> (1, 94) (1*P*1 => 1*P)
            # labels reshape: (1, 94) -> (94, ) (1*N => N,)
            # beta_nn_input = beta_nn_input.squeeze(0).T

            factor_nn_input = factor_nn_input.squeeze(0).T
            labels = labels.squeeze(0)
            output = self.forward(factor_nn_input)
            
            loss = self.criterion(output, labels)
            
            # Apply L1 regularization
            lambda_reg = 0.01
            l1_norm = sum(p.abs().sum() for p in self.parameters())
            loss += lambda_reg * l1_norm

            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            if i % 100 == 0:
                # print(f'Batches: {i}, loss: {loss.item()}')
                pass

        return epoch_loss / len(self.train_dataloader)
    
    def __valid_one_epoch(self):
        epoch_loss = 0.0

        for i, (factor_nn_input, labels) in enumerate(self.valid_dataloader):
            # beta_nn_input reshape: (1, 94, 94) -> (94, 94) (1*P*N => N*P)
            # factor_nn_input reshape: (1, 94, 1) -> (1, 94) (1*P*1 => 1*P)
            # labels reshape: (1, 94) -> (94, ) (1*N => N,)
            # beta_nn_input = beta_nn_input.squeeze(0).T
            factor_nn_input = factor_nn_input.squeeze(0).T
            labels = labels.squeeze(0)

            output = self.forward(factor_nn_input)
            loss = self.criterion(output, labels)
        
                
            # loss = self.criterion(output, labels)
            epoch_loss += loss.item()
        
        return epoch_loss / len(self.valid_dataloader)
    
    def train_model(self):
        if 'saved_models' not in os.listdir('./'):
            os.mkdir('saved_models')
        
        self.train_dataloader = self.dataloader(self.train_period)
        self.valid_dataloader = self.dataloader(self.valid_period)
        self.test_dataloader = self.dataloader(self.test_period)
        
        min_error = np.Inf
        no_update_steps = 0
        valid_loss = []
        train_loss = []
        for i in range(MAX_EPOCH):
            # print(f'Epoch {i}')
            self.train()
            train_error = self.__train_one_epoch()
            train_loss.append(train_error)
            
            self.eval()
            # valid and early stop
            with torch.no_grad():
                valid_error = self.__valid_one_epoch()
            # Print train and valid loss
   

            valid_loss.append(valid_error)
            if valid_error < min_error:
                min_error = valid_error
                no_update_steps = 0
                print(f"Valid loss: {valid_error}")
                # save model
                torch.save(self.state_dict(), f'./saved_models/{self.name}.pt')
            else:
                no_update_steps += 1
            
            if no_update_steps > 20: # early stop, if consecutive 3 epoches no improvement on validation set
                print(f'Early stop at epoch {i}')
                break
            # load from (best) saved model
        self.load_state_dict(torch.load(f'./saved_models/{self.name}.pt'))
        return train_loss, valid_loss
    