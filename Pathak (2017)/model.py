import torch.nn as nn
from torch.distributions import Categorical
import torch
import torch.optim as optim
import torch.nn.functional as F

def Conv_Head():
    return nn.Sequential(
                    nn.Conv2d(3, 32, 3, stride=2, padding=1),
                    nn.ELU(),
                    nn.Conv2d(32, 32, 3, stride=2, padding=1),
                    nn.ELU(),
                    nn.Conv2d(32, 32, 3, stride=2, padding=1),
                    nn.ELU(),
                    nn.Conv2d(32, 32, 3, stride=2, padding=1),
                    nn.ELU()
                )

class Inverse_Model(nn.Module):
    def __init__(self, action_dim):
        super().__init__()

        self.feature_extractor = Conv_Head()

        self.dense1 = nn.Linear(288+288, 256)
        self.dense2 = nn.Linear(256, action_dim)

    def forward(self, state, nxt_state):
        state = self.get_feature(state).flatten(start_dim=1)
        nxt_state = self.get_feature(nxt_state).flatten(start_dim=1)
        
        x = F.relu(self.dense1(torch.cat((state, nxt_state), dim=1)))
        x = self.dense2(x)

        return state, nxt_state, Categorical(x)
    
    def get_feature(self, x):
        return self.feature_extractor(x)

class Forward_Model(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
    
        self.dense1 = nn.Linear(288 + 1, 256)
        self.dense2 = nn.Linear(256, 288)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))

        return x

class A2C_Model(nn.Module):
    def __init__(self, action_dim):
        super().__init__()

        self.conv = Conv_Head()
        self.dense = nn.Linear(288, 256)
        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)
    
    def forward(self, x):
        x = self.conv(x).flatten(start_dim=1)
        x = F.elu(self.dense(x))

        return Categorical(logits=self.actor(x)), self.critic(x)

class ICM(nn.Module):
    def __init__(self, action_dim):
        super().__init__()

        self.inverse = Inverse_Model(action_dim)
        self.forward_ = Forward_Model(action_dim)
    
    def forward(self, state, act, nxt_state):
        state_feature, nxt_state_feature, act_dist = self.inverse(state, nxt_state)
        
        input_ = torch.cat((act.unsqueeze(1).float(), state_feature.detach()), dim=1)
        feature_prediction = self.forward_(input_)

        curiosity = 0.5 * (feature_prediction - nxt_state_feature).pow(2).sum(dim=1)
        return curiosity, act_dist