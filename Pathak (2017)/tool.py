import torch
import cv2
import scipy

class buffer():
    def __init__(self, size):
        self.reset()

    def reset(self):
        self.act = []
        self.obs = []
        self.nxt_obs = []
        self.obs_feature = []
        self.nxt_obs_feature = []
        self.rwd = []

    def as_tensor(self):
        buff = {
            'act' : torch.as_tensor(self.act, dtype=torch.int32),
            'obs' : torch.stack(self.obs),
            'nxt_obs' : torch.stack(self.nxt_obs),
            'obs_feature' : torch.cat(self.obs_feature),
            'nex_obs_feature' : torch.cat(self.nxt_obs_feature),
            'rwd' : torch.as_tensor(self.rwd, dtype=torch.float32)
        }

        return buff

def discount(x, gamma):
    discounted_rwds = []
    ret = 0
    for r in reversed(x):
        ret = ret*gamma + r
        discounted_rwds.insert(0, ret)
    
    return torch.stack(discounted_rwds).unsqueeze(1)

def preprocess(obs):
    normalized = cv2.resize(obs, (42, 42), interpolation=cv2.INTER_AREA) / 256
    return torch.as_tensor(normalized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
