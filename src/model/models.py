import numpy as np
import torch
import torch.nn as nn


def create_model(args, day_length, action_dim):
    if args.model == 'tcn':
        model = CNN_tcn(args, day_length, action_dim)

    elif args.model == 'EIIE':
        model = CNN_EIIE(args, day_length, action_dim)

    else:
        raise NotImplementedError

    return model


class CNN_tcn(nn.Module):
    def __init__(self, args, day_length, action_dim):
        super(CNN_tcn, self).__init__()
        self.alpha = 1
        self.cvks1 = 3
        self.cvks2 = day_length
        self.in_ch1 = 4
        self.out_ch1 = int(4*self.alpha)
        self.out_ch2 = int(16*self.alpha)
        self.out_ch3 = int(1*self.alpha)
        self.action_dim = action_dim
        self.relu = nn.ReLU()
        self.bool = False
        
        ##### layer1 #####
        self.DConv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_ch1, out_channels=self.out_ch1,
                      kernel_size=(1, self.cvks1), stride=1,
                      padding=(0, int((self.cvks1+1)/2)),
                      bias=False, dilation=2),
            nn.LayerNorm(normalized_shape=[self.out_ch1, self.action_dim, day_length], elementwise_affine=self.bool),
            nn.Conv2d(in_channels=self.out_ch1, out_channels=self.out_ch1,
                      kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.GELU()
        )
        
        self.DConv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_ch1, out_channels=self.out_ch1,
                      kernel_size=(1, self.cvks1), stride=1,
                      padding=(0, int((self.cvks1+1)/2)),
                      bias=False, dilation=2),
            nn.LayerNorm(normalized_shape=[self.out_ch1, self.action_dim, day_length], elementwise_affine=self.bool),
            nn.Conv2d(in_channels=self.out_ch1, out_channels=self.out_ch1,
                      kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.GELU()
        )
        
        self.DConv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_ch1, out_channels=self.out_ch1,
                      kernel_size=(1, self.cvks1), stride=1,
                      padding=(0, int((self.cvks1+1)/2)),
                      bias=False, dilation=2),
            nn.LayerNorm(normalized_shape=[self.out_ch1, self.action_dim, day_length], elementwise_affine=self.bool),
            nn.Conv2d(in_channels=self.out_ch1, out_channels=self.out_ch1,
                      kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.GELU()
        )
        
        self.Conv = nn.Sequential(        
            nn.LayerNorm(normalized_shape=[self.out_ch1 * 3, self.action_dim, day_length], elementwise_affine=self.bool),
            nn.Conv2d(in_channels=self.out_ch1*3, out_channels=self.out_ch1,
                      kernel_size=1, stride=1,
                      padding=0, bias=False),
        )
 
        ##### layer2 #####
        self.Conv2 = nn.Sequential(      
            nn.Conv2d(in_channels=self.out_ch1, out_channels=self.out_ch2,
                      kernel_size=(1, self.cvks2), stride=1, bias=False),
            nn.LayerNorm(normalized_shape=[self.out_ch2, self.action_dim, 1], elementwise_affine=self.bool),
        )
        
        #### layer3 #####
        self.Conv3 = nn.Sequential(        
            nn.Conv2d(in_channels=self.out_ch2+1, out_channels=self.out_ch3,
                      kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.LayerNorm(normalized_shape=[self.out_ch3, self.action_dim, 1], elementwise_affine=self.bool),
        )
        
    def forward(self, s, w):
        s = s.permute(0, 3, 1, 2)                           # inputs
        x1 = self.DConv1(s)                                 # DC Layer 1
        x2 = self.DConv2(s + x1)                            # DC Layer 2
        x3 = self.DConv3(s + x1 + x2)                       # DC Layer 3

        x = torch.cat((x1, x2, x3), 1)                      # concate
        x = self.Conv(x)
        x = self.Conv2(x)                                   # Layer 2
        x = torch.cat((x, w.unsqueeze(1).unsqueeze(-1)), 1) # previous weight
        x = self.Conv3(x)                                   # Layer 3    
        x = x.squeeze(3).squeeze(1)
        return x


class CNN_EIIE(nn.Module): # Ensemble of Identical Independent Evaluators
    def __init__(self, args, day_length, action_dim):
        super(CNN_EIIE, self).__init__()
        self.alpha = 1
        self.cvks1 = 3
        self.cvks2 = day_length
        self.in_ch1 = 3
        self.out_ch1 = int(2 * self.alpha)
        self.out_ch2 = int(20 * self.alpha)
        self.out_ch3 = 1
        self.action_dim = action_dim
        
        ##### layer1 #####
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_ch1, out_channels=self.out_ch1,
                      kernel_size=(1, self.cvks1), stride=1,
                      padding=(0, int((self.cvks1-1)/2))),
            nn.BatchNorm2d(num_features=self.out_ch1, eps=1e-05, momentum=0.1, 
                            affine=True, track_running_stats=True),
            nn.ReLU()
        )
        
        ##### layer2 #####
        self.Conv2 = nn.Sequential(        
            nn.Conv2d(in_channels=self.out_ch1, out_channels=self.out_ch2,
                      kernel_size=(1, self.cvks2), stride=1),
            nn.BatchNorm2d(num_features=self.out_ch2, eps=1e-05, momentum=0.1, 
                           affine=True, track_running_stats=True),
            nn.ReLU()
        )
        
        ##### layer3 ##### 
        self.Conv3 = nn.Sequential(         
            nn.Conv2d(in_channels=self.out_ch2+1, out_channels=self.out_ch3,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=self.out_ch3, eps=1e-05, momentum=0.1, 
                           affine=True, track_running_stats=True),
        )
        
    def forward(self, s, w):
        s = s.permute(0, 3, 1, 2)                               # inputs
        x = self.Conv1(s[:, 1:, :, :])                          # Layer 1
        x = self.Conv2(x)                                       # Layer 2
        x = torch.cat((x, w.unsqueeze(1).unsqueeze(-1)), 1)
        x = self.Conv3(x)                                       # Layer 3    
        x = x.squeeze(3).squeeze(1)
        return x


class FNN(nn.Module):
    def __init__(self, args, action_dim, hidden_dim1, hidden_dim2, output_dim):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(action_dim, hidden_dim1)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim1)
        self.fc3.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(hidden_dim1, output_dim)
        self.out.weight.data.normal_(0, 0.1)  # initialization
        self.relu = nn.ReLU()

    def forward(self, s):             
        x = self.relu(self.fc1(s)) 
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.out(x)
        return x


class CriticFC(nn.Module):
    def __init__(self, action_dim, hidden_dim, output_dim):
        super(CriticFC, self).__init__()
        self.fca = nn.Linear(action_dim, hidden_dim)
        self.fcs = nn.Linear(action_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, s, weight):
        v1 = self.fcs(s)
        v2 = self.fca(weight)
        value = self.out(self.relu(v1 + v2))
        return value
    
    
