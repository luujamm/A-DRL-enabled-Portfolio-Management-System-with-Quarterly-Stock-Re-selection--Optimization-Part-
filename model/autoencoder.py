import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data


def MA(data, p):
    ma = np.zeros(data.shape)
    for i in range(p):
        ma += data[:, i:] / p
        data = np.concatenate((data, np.zeros((data.shape[0], 1))), axis=1)
    return ma[:, : -p + 1]

def feature_convert(train_history, train_dating, p=5):
    num_stocks, num_days, _ = train_history.shape
    #print('data:', train_history.shape)
    # Normalized hcl
    
    train_data = train_history[:, 1:, :] / train_history[:, :-1, -1:]             #(num, d, 4)
    
    # ATR
    TR = np.concatenate((train_history[:, :, :1] - train_history[:, :, 2: 3],
                        np.abs(train_history[:, :, :1] - train_history[:, :, 1: 2])), 2)
    TR = np.concatenate((TR, np.abs(train_history[:, :, 2: 3] - train_history[:, :, 1: 2])), 2)
    TR = np.amax(TR, axis=2)                                                      #TR = max((H-L), abs(H-C), abs(C-L))
    ATR = MA(TR, p)   
    #print('ATR:', ATR.shape)                                                            #(num, d-(p-1))

    FI = ATR.copy() #Financial indicators
    
    # CCI = (typical_price - ma) / (0.015 * mean_deviation)
    # typical_price = sum((H + C + L) / 3)
    typical_price = MA(np.mean(train_history[:, :, 1:], axis=2), p) * p                              #(num, d-(p-1))
    typical_price_ma = MA(typical_price, p)                                                             #(num, d-2(p-1))
    typical_price_md = MA(np.abs(typical_price[:, : -p+1] - typical_price_ma), p)                       #(num, d-3(p-1))
    CCI = (typical_price[:, : -2 * (p-1)] - typical_price_ma[:, : -(p-1)]) / (0.015 * typical_price_md) #(num, d-3(p-1))
    #print('CCI:', CCI.shape)

    # Demand index
    K = np.concatenate((train_history[:, :-1, :1], train_history[:, 1:, :1]), 2)
    K = np.concatenate((K, train_history[:, :-1, :3]), 2)
    K = np.concatenate((K, train_history[:, 1:, :3]), 2)
    K = 3 * train_history[:, p:, 2] / MA(np.amax(K, axis=2) - np.amin(K, axis=2), p)                     #(num, d-p)
    DI = K * (train_history[:, p:, 2] / train_history[:, p-1: -1, 2] - 1)
    for i in range(len(DI)):
        for j in range(len(DI[i])):
            if abs(DI[i][j]) > 1:
                DI[i][j] = 1 / DI[i][j]
    #print('DI: ', DI.shape)
    
    
    # Dynamic momentum index
    std = np.zeros((num_stocks, 1))
    for i in range(5, num_days):
        std = np.concatenate((std, np.std(train_history[:, i - 5: i, 2], axis=1).reshape(-1, 1)), axis=1)
        
    std_10ma = MA(std, 10)  
    Vi = std[:, 9:] / std_10ma 
    TD = np.int_(14 / (Vi + 1e-3))[:, 16:]
    TD[TD > 30] = 30
    TD[TD < 5] = 5
    #print(train_history[:,:20])
    change = train_history[:, 1:, 2] - train_history[:, : -1, 2]
    RS = np.zeros(TD.shape)
    for i in range(num_stocks):
        for j in range(TD.shape[1]):
            change_cut = change[i, j + 30 - TD[i][j]: j + 30]
            avg_gain = change_cut[change_cut > 0]
            avg_loss = np.abs(change_cut[change_cut < 0])
            if len(avg_loss) == 0:
                avg_loss = np.array([1e-2])
            if len(avg_gain) == 0:
                avg_gain = np.array([1e-2])
            RS[i][j] = np.mean(avg_gain) / np.mean(avg_loss)
    DMI = 100 * (1 - 1 / (1 + RS))
    #print('DMI:', DMI.shape)
    
    # EMA
    EMA = MA(train_history[:, :p, 2], p)
    s = 2 / (1 + p)
    for i in range(p, num_days):
        new = train_history[:, i, 2] * s + EMA[:, -1] * (1 - s)
        EMA = np.concatenate((EMA, new.reshape(-1, 1)), axis=1)
    #print('EMA:', EMA.shape)
    
    # Momentum
    MTM = train_history[:, p:, 2] - train_history[:, :-p, 2]
    #print('MTM:', MTM.shape)
    
    l = DMI.shape[1]
    FI = np.concatenate((FI[:, -l:].reshape(num_stocks,-1,1), CCI[:, -l:].reshape(num_stocks,-1,1)), axis=2)
    FI = np.concatenate((FI, DI[:, -l:].reshape(num_stocks,-1,1)), axis=2)
    FI = np.concatenate((FI, DMI.reshape(num_stocks,-1,1)), axis=2)
    FI = np.concatenate((FI, EMA[:, -l:].reshape(num_stocks,-1,1)), axis=2)
    FI = np.concatenate((FI, MTM[:, -l:].reshape(num_stocks,-1,1)), axis=2)
    
    #print(train_data.shape)
    
    
    FI = FI[:, 1:, :] / (FI[:, :-1, :] + 1e-8)
    
    train_data = np.concatenate((train_data[:, -l+1:, :], FI), axis=2)
    
    
    
    train_dating = train_dating[-l+1:]
    return train_data , train_dating


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.input_dim = 10
        self.out_dim = 4
        self.input = nn.Linear(self.input_dim, self.input_dim - 1)
        self.dense1 = nn.Linear(self.input_dim - 1, self.input_dim - 2)
        self.dense2 = nn.Linear(self.input_dim - 2, self.out_dim)
        self.dense3 = nn.Linear(self.out_dim, self.out_dim)
        self.fc = nn.Linear(self.out_dim, self.out_dim)
        self.tanh = nn.Tanh()
       
        
    def forward(self, input):
        x = self.tanh(self.input(input))
        x = self.tanh(self.dense1(x))
        x = self.tanh(self.dense2(x))
        extracted_x = self.tanh(self.dense3(x)) #Extracted Features
        x = self.tanh(self.dense3(extracted_x))
        x = self.tanh(self.dense3(x))
        x = self.fc(x)

        return x, extracted_x

class Autoencoder():
    def __init__(self, args):
        self.device = args.device
        self.num_epoch = args.pretrain_epoch
        self.pretrain_batch_size = args.pretrain_batch_size
        self.model = AE().to(self.device)
    
    def pretrain(self, train_history, train_dating):
        model = self.model
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr = 0.0001)
        data, _ = feature_convert(train_history, train_dating)
        
        use_data = torch.FloatTensor(data.reshape(-1, 10))
        
        train_size = int(0.8*len(use_data))
        train_data = use_data[:train_size, :].to(self.device)
        
        train_label = train_data[:, :4]
        
        val_data = use_data[train_size:, :].to(self.device)
        val_label = val_data[:, :4]
        train_dataset = Data.TensorDataset(train_data, train_label)
        train_loader = Data.DataLoader(train_dataset, self.pretrain_batch_size)
        
        print('Pretrain Start')
        for t in range(self.num_epoch):
            model.train()
            for x_train, y_train in train_loader:   
                optimizer.zero_grad()
                y_train_pred, _ = model(x_train)
                loss = loss_fn(y_train_pred, y_train)
                loss.backward()
                optimizer.step()
               
                
            
            with torch.no_grad():
                model.eval()
                y_val_pred, _ = model(val_data)
                
                
            val_loss = loss_fn(y_val_pred, val_label)

            if (t+1)%100 == 0:
                
                print('Epoch %d: train loss = %f, val loss = %f'%(t+1, loss.item(), val_loss.item()))
        
    
    def extract(self, state):
        
        state = torch.FloatTensor(state).to(self.device)
        
        s, extracted_state = self.model(state)
        
        return extracted_state.detach().cpu().numpy()

    def save(self, path):
        torch.save({'AE': self.model.state_dict()}, path)
    
    def load(self, path):
        model = torch.load(path)
        self.model.load_state_dict(model['AE'])