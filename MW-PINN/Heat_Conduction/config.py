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
        self.n_collocation = 10000
        self.n_validation = 1000
        self.n_initial = 500
        self.n_boundary = 500
        self.n_test = 200
        
        self.x_lower = -1
        self.x_upper = 1
        self.t_lower = 0
        self.t_upper = 1
        
        self.device = device
    
    def generate_training_points(self):
        sampler = qmc.Sobol(d = 2, scramble = True, seed = 501)
        sobol_sequence_collocation = sampler.random(n = self.n_collocation)
        sobol_sequence_boundary = sampler.random(n = self.n_boundary)

        x_collocation = torch.tensor(sobol_sequence_collocation[:,0].flatten()*(self.x_upper - self.x_lower) + self.x_lower).float().to(device)
        t_collocation = torch.tensor(sobol_sequence_collocation[:,1].flatten()*(self.t_upper - self.t_lower) + self.t_lower).float().to(device)

        x_ic = torch.tensor(sobol_sequence_boundary[:,0].flatten()*(self.x_upper - self.x_lower) + self.x_lower).float().to(device)
        t_ic = self.t_lower * torch.ones(self.n_boundary).to(self.device)
        
        t_bc = torch.tensor(sobol_sequence_boundary[:,1].flatten()*(self.t_upper - self.t_lower) + self.t_lower).float().to(device)
        x_bc_left = self.x_lower * torch.ones(self.n_boundary).to(self.device)
        x_bc_right = self.x_upper * torch.ones(self.n_boundary).to(self.device)

        x_validation = (torch.rand(self.n_validation) * (self.x_upper - self.x_lower) + self.x_lower).to(device)
        t_validation = (torch.rand(self.n_validation) * (self.t_upper - self.t_lower) + self.t_lower).to(device)

        xtest = torch.linspace(self.x_lower, self.x_upper, self.n_test)
        ttest = torch.linspace(self.t_lower, self.t_upper, self.n_test)
            
        x_grid, t_grid = torch.meshgrid(xtest, ttest, indexing='ij')
        x_test = x_grid.reshape(-1)
        t_test = t_grid.reshape(-1)
        
        return {
            'domain': (self.x_lower, self.x_upper, self.t_lower, self.t_upper),  
            'collocation': (self.n_collocation, x_collocation, t_collocation),
            'validation': (x_validation, t_validation),
            'initial': (x_ic, t_ic),
            'boundary': (t_bc, x_bc_left, x_bc_right),
            'test': (self.n_test, x_test, t_test)
        }

config = DataConfig()
points = config.generate_training_points()

x_lower, x_upper, t_lower, t_upper = points['domain']
n_collocation, x_collocation, t_collocation = points['collocation']
x_validation, t_validation = points['validation']
x_ic, t_ic = points['initial']
t_bc, x_bc_left, x_bc_right = points['boundary']
n_test, x_test, t_test = points['test']
