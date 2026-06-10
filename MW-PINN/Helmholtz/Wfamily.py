import torch
from config import *
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.meta_wavelet import HermiteBasis

def build_family():
    Jx = torch.arange(-4.0, 6.0)
    Jy = torch.arange(-4.0, 6.0)
    a = 0.5
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
X_bc_left = compute_inputs(x_bc_left, family_x)
X_bc_right = compute_inputs(x_bc_right, family_x)
Y_bc_lr = compute_inputs(y_bc, family_y)

X_bc_tb = compute_inputs(x_bc, family_x)
Y_bc_bottom = compute_inputs(y_bc_bottom, family_y)
Y_bc_top = compute_inputs(y_bc_top, family_y)

# Validation inputs
X_val = compute_inputs(x_validation, family_x)
Y_val = compute_inputs(y_validation, family_y)

# Test inputs
X_test = compute_inputs(x_test.to(device), family_x)
Y_test = compute_inputs(y_test.to(device), family_y)

class WaveletEvaluator:
    def __init__(self, N_H=2):
        self.basis = HermiteBasis(N_H=N_H).to(device)
        self.jx = family_x[:, 0:1] # [N_x, 1]
        self.jy = family_y[:, 0:1] # [N_y, 1]
        
    def evaluate(self, c_flat, b):
        c = c_flat.view(N_x, N_y)
        
        # Collocation
        psi_X = self.basis.evaluate(X_coll, derivative=0) 
        psi_Y = self.basis.evaluate(Y_coll, derivative=0)  
        d2psi_X = self.basis.evaluate(X_coll, derivative=2) 
        d2psi_Y = self.basis.evaluate(Y_coll, derivative=2)
        
        W_x = psi_X.T      # [N_coll, N_x]
        W_y = psi_Y.T      # [N_coll, N_y]
        DW2_x = (self.jx**2 * d2psi_X).T 
        DW2_y = (self.jy**2 * d2psi_Y).T 
        
        u = torch.sum((W_x @ c) * W_y, dim=1) + b
        u_xx = torch.sum((DW2_x @ c) * W_y, dim=1)
        u_yy = torch.sum((W_x @ c) * DW2_y, dim=1)
        
        # BC Left
        W_bcl_x = self.basis.evaluate(X_bc_left, derivative=0).T
        W_bclr_y = self.basis.evaluate(Y_bc_lr, derivative=0).T
        u_bcl = torch.sum((W_bcl_x @ c) * W_bclr_y, dim=1) + b
        
        # BC Right
        W_bcr_x = self.basis.evaluate(X_bc_right, derivative=0).T
        u_bcr = torch.sum((W_bcr_x @ c) * W_bclr_y, dim=1) + b
        
        # BC Bottom
        W_bctb_x = self.basis.evaluate(X_bc_tb, derivative=0).T
        W_bcb_y = self.basis.evaluate(Y_bc_bottom, derivative=0).T
        u_bcb = torch.sum((W_bctb_x @ c) * W_bcb_y, dim=1) + b

        # BC Top
        W_bct_y = self.basis.evaluate(Y_bc_top, derivative=0).T
        u_bct = torch.sum((W_bctb_x @ c) * W_bct_y, dim=1) + b
        
        return u, u_xx, u_yy, u_bcl, u_bcr, u_bcb, u_bct
        
    def evaluate_val(self, c_flat, b):
        c = c_flat.view(N_x, N_y)
        W_x = self.basis.evaluate(X_val, derivative=0).T
        W_y = self.basis.evaluate(Y_val, derivative=0).T
        return torch.sum((W_x @ c) * W_y, dim=1) + b

    def evaluate_test(self, c_flat, b):
        c = c_flat.view(N_x, N_y)
        W_x = self.basis.evaluate(X_test, derivative=0).T
        W_y = self.basis.evaluate(Y_test, derivative=0).T
        return torch.sum((W_x @ c) * W_y, dim=1) + b
