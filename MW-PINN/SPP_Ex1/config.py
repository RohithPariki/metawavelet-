import torch
from scipy.stats import qmc
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

global device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
torch.manual_seed(101)

class DataConfig:
    def __init__(self):
        self.n_collocation = 1000
        self.n_validation = 1000
        self.n_boundary = 100
        self.ntest = 10000
        
        self.x_lower = 0
        self.x_upper = 1
        self.device = device
    
    def generate_training_points(self):
        sampler = qmc.Sobol(d = 1, scramble = True, seed = 501)
        sobol_sequence = sampler.random(n = self.n_collocation)* (self.x_upper - self.x_lower) + self.x_lower 
        x_collocation = torch.tensor(sobol_sequence.flatten()).float().to(device)
   
        x_validation = (torch.rand(self.n_validation) * (self.x_upper - self.x_lower) + self.x_lower).to(device)

        x_bc_left = self.x_lower * torch.ones(self.n_boundary).to(self.device)
        x_bc_right = self.x_upper * torch.ones(self.n_boundary).to(self.device)

        x_test = torch.linspace(self.x_lower, self.x_upper, self.ntest).to(device)
        
        return {
            'domain': (self.x_lower, self.x_upper),  
            'collocation': (self.n_collocation, x_collocation),
            'validation': (x_validation),
            'boundary': (x_bc_left, x_bc_right),
            'test': (x_test)
        }

config = DataConfig()
points = config.generate_training_points()

x_lower, x_upper = points['domain']
n_collocation, x_collocation = points['collocation']
x_validation = points['validation']
x_bc_left, x_bc_right = points['boundary']
x_test = points['test']
