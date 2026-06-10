import torch
import numpy as np
import matplotlib.pyplot as plt

global device, Re
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
Re = 100
torch.manual_seed(121)

class DataConfig:
    def __init__(self):
        self.n_collocation = 40000
        self.n_boundary = 1000
        self.n_test = 100
        
        self.x_lower = 0
        self.x_upper = 1
        self.y_lower = 0
        self.y_upper = 1
        
        self.device = device
    
    def generate_training_points(self):
        x_collocation = (torch.rand(self.n_collocation) * (self.x_upper - self.x_lower) + self.x_lower).to(device)
        y_collocation = (torch.rand(self.n_collocation) * (self.y_upper - self.y_lower) + self.y_lower).to(device)

        # Boundary condition points
        x_bc = (torch.rand(self.n_boundary) * (self.x_upper - self.x_lower) + self.x_lower).to(self.device)
        y_bc_lower = self.y_lower * torch.ones(self.n_boundary).to(self.device)
        
        x_bc_upper = (torch.rand(self.n_boundary) * (self.x_upper - self.x_lower) + self.x_lower).to(self.device)
        y_bc_upper = self.y_upper * torch.ones(self.n_boundary).to(self.device)
        
        y_bc = (torch.rand(self.n_boundary) * (self.y_upper - self.y_lower) + self.y_lower).to(self.device)
        x_bc_left = self.x_lower * torch.ones(self.n_boundary).to(self.device)
        x_bc_right = self.x_upper * torch.ones(self.n_boundary).to(self.device)

        vel_ref = torch.tensor(np.loadtxt("ref_vel.csv").astype(np.float32)).to(device)
        
        xtest = torch.linspace(self.x_lower, self.x_upper, self.n_test).to(device)
        ytest = torch.linspace(self.y_lower, self.y_upper, self.n_test).to(device)
        x_grid, y_grid = torch.meshgrid(xtest, ytest, indexing='ij')
        x_test_grid = x_grid.reshape(-1)
        y_test_grid = y_grid.reshape(-1)
        
        return {
            'domain': (self.x_lower, self.x_upper, self.y_lower, self.y_upper),
            'collocation': (self.n_collocation, x_collocation, y_collocation),
            'boundary': (x_bc, x_bc_upper, y_bc_lower, y_bc_upper, y_bc, x_bc_left, x_bc_right),
            'test': (self.n_test, x_test_grid, y_test_grid),
            'ref': (vel_ref)
        }

config = DataConfig()
points = config.generate_training_points()

x_lower, x_upper, y_lower, y_upper = points['domain']
len_collocation, x_collocation, y_collocation = points['collocation']
x_bc, x_bc_upper, y_bc_lower, y_bc_upper, y_bc, x_bc_left, x_bc_right = points['boundary']
n_test, x_test, y_test = points['test']
vel_ref = points['ref']
