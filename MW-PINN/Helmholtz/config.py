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
        self.n_boundary = 1000
        self.n_test = 200
        
        self.x_lower = -1
        self.x_upper = 1
        self.y_lower = -1
        self.y_upper = 1
        
        self.device = device
    
    def generate_training_points(self):
        sampler = qmc.Sobol(d = 2, scramble = True, seed = 501)
        sobol_sequence_collocation = sampler.random(n = self.n_collocation)

        x_collocation = torch.tensor(sobol_sequence_collocation[:,0].flatten()*(self.x_upper - self.x_lower) + self.x_lower).float().to(device)
        y_collocation = torch.tensor(sobol_sequence_collocation[:,1].flatten()*(self.y_upper - self.y_lower) + self.y_lower).float().to(device)

        # Boundary condition points
        x_bc = (torch.rand(self.n_boundary) * (self.x_upper - self.x_lower) + self.x_lower).to(self.device)
        y_bc_bottom = self.y_lower * torch.ones(self.n_boundary).to(self.device)
        y_bc_top = self.y_upper * torch.ones(self.n_boundary).to(self.device)
        
        y_bc = (torch.rand(self.n_boundary) * (self.y_upper - self.y_lower) + self.y_lower).to(self.device)
        x_bc_left = self.x_lower * torch.ones(self.n_boundary).to(self.device)
        x_bc_right = self.x_upper * torch.ones(self.n_boundary).to(self.device)

        # Validation points
        x_validation = (torch.rand(self.n_validation) * (self.x_upper - self.x_lower) + self.x_lower).to(device)
        y_validation = (torch.rand(self.n_validation) * (self.y_upper - self.y_lower) + self.y_lower).to(device)

        # Testing and Plotting points
        xtest = torch.linspace(self.x_lower, self.x_upper, self.n_test).to(device)
        ytest = torch.linspace(self.y_lower, self.y_upper, self.n_test).to(device)
            
        x_grid, y_grid = torch.meshgrid(xtest, ytest, indexing='ij')
        x_test = x_grid.reshape(-1)
        y_test = y_grid.reshape(-1)
        
        return {
            'domain': (self.x_lower, self.x_upper, self.y_lower, self.y_upper),  
            'collocation': (self.n_collocation, x_collocation, y_collocation),
            'validation': (x_validation, y_validation),
            'boundary': (x_bc, y_bc_bottom, y_bc_top, y_bc, x_bc_left, x_bc_right),
            'test': (self.n_test, x_test, y_test)
        }

config = DataConfig()
points = config.generate_training_points()

x_lower, x_upper, y_lower, y_upper = points['domain']
n_collocation, x_collocation, y_collocation = points['collocation']
x_validation, y_validation = points['validation']
x_bc, y_bc_bottom, y_bc_top, y_bc, x_bc_left, x_bc_right = points['boundary']
n_test, x_test, y_test = points['test']
