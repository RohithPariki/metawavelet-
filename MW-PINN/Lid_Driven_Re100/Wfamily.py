import torch
from config import *
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.meta_wavelet import HermiteBasis

def build_family():
    Jx = torch.arange(-3.0, 5.0)
    Jy = torch.arange(-3.0, 5.0)
    a = 1
    family_x = torch.tensor([(2**jx, kx) for jx in Jx for kx in range(int(torch.floor((x_lower-a)*2**(jx))), int(torch.ceil((x_upper+a)*2**(jx))) + 1)])
    family_y = torch.tensor([(2**jy, ky) for jy in Jy for ky in range(int(torch.floor((y_lower-a)*2**(jy))), int(torch.ceil((y_upper+a)*2**(jy))) + 1)])
    return family_x.to(device), family_y.to(device)

family_x, family_y = build_family()
N_x = len(family_x)
N_y = len(family_y)
len_family = N_x * N_y
print("family_len: ", len_family)

def compute_inputs(pts, family_dim):
    j = family_dim[:, 0]
    k = family_dim[:, 1]
    return j[:, None] * pts[None, :] - k[:, None]

# Collocation inputs
X_coll = compute_inputs(x_collocation, family_x)
Y_coll = compute_inputs(y_collocation, family_y)

# Boundary inputs
X_bc = compute_inputs(x_bc, family_x)
X_bc_upper = compute_inputs(x_bc_upper, family_x)
X_bc_left = compute_inputs(x_bc_left, family_x)
X_bc_right = compute_inputs(x_bc_right, family_x)

Y_bc = compute_inputs(y_bc, family_y)
Y_bc_lower = compute_inputs(y_bc_lower, family_y)
Y_bc_upper = compute_inputs(y_bc_upper, family_y)

# Test inputs
X_test = compute_inputs(x_test, family_x)
Y_test = compute_inputs(y_test, family_y)

class WaveletEvaluator:
    def __init__(self, N_H=2):
        self.basis = HermiteBasis(N_H=N_H).to(device)
        self.jx = family_x[:, 0:1] # [N_x, 1]
        self.jy = family_y[:, 0:1] # [N_y, 1]
        
    def evaluate_navier_stokes(self, c_flat, b, is_u=False, is_v=False, is_p=False):
        c = c_flat.view(N_x, N_y)
        
        psi_X = self.basis.evaluate(X_coll, derivative=0) 
        psi_Y = self.basis.evaluate(Y_coll, derivative=0)
        
        W_x = psi_X.T
        W_y = psi_Y.T
        
        u = torch.sum((W_x @ c) * W_y, dim=1) + b
        
        if is_p:
            dpsi_X = self.basis.evaluate(X_coll, derivative=1)
            dpsi_Y = self.basis.evaluate(Y_coll, derivative=1)
            DW_x = (self.jx * dpsi_X).T
            DW_y = (self.jy * dpsi_Y).T
            u_x = torch.sum((DW_x @ c) * W_y, dim=1)
            u_y = torch.sum((W_x @ c) * DW_y, dim=1)
            return u, u_x, u_y
            
        d2psi_X = self.basis.evaluate(X_coll, derivative=2) 
        d2psi_Y = self.basis.evaluate(Y_coll, derivative=2)
        dpsi_X = self.basis.evaluate(X_coll, derivative=1)
        dpsi_Y = self.basis.evaluate(Y_coll, derivative=1)
        
        DW_x = (self.jx * dpsi_X).T
        DW_y = (self.jy * dpsi_Y).T
        DW2_x = (self.jx**2 * d2psi_X).T 
        DW2_y = (self.jy**2 * d2psi_Y).T 
        
        u_x = torch.sum((DW_x @ c) * W_y, dim=1)
        u_y = torch.sum((W_x @ c) * DW_y, dim=1)
        u_xx = torch.sum((DW2_x @ c) * W_y, dim=1)
        u_yy = torch.sum((W_x @ c) * DW2_y, dim=1)
        
        return u, u_x, u_y, u_xx, u_yy
        
    def evaluate_bc(self, c_flat, b):
        c = c_flat.view(N_x, N_y)
        
        W_bc_x = self.basis.evaluate(X_bc, derivative=0).T
        W_bc_low_y = self.basis.evaluate(Y_bc_lower, derivative=0).T
        u_bc_low = torch.sum((W_bc_x @ c) * W_bc_low_y, dim=1) + b
        
        W_bcup_x = self.basis.evaluate(X_bc_upper, derivative=0).T
        W_bcup_y = self.basis.evaluate(Y_bc_upper, derivative=0).T
        u_bc_up = torch.sum((W_bcup_x @ c) * W_bcup_y, dim=1) + b
        
        W_bcl_x = self.basis.evaluate(X_bc_left, derivative=0).T
        W_bc_y = self.basis.evaluate(Y_bc, derivative=0).T
        u_bc_left = torch.sum((W_bcl_x @ c) * W_bc_y, dim=1) + b
        
        W_bcr_x = self.basis.evaluate(X_bc_right, derivative=0).T
        u_bc_right = torch.sum((W_bcr_x @ c) * W_bc_y, dim=1) + b
        
        return u_bc_low, u_bc_up, u_bc_left, u_bc_right
        
    def evaluate_test(self, c_flat, b):
        c = c_flat.view(N_x, N_y)
        W_x = self.basis.evaluate(X_test, derivative=0).T
        W_y = self.basis.evaluate(Y_test, derivative=0).T
        return torch.sum((W_x @ c) * W_y, dim=1) + b
