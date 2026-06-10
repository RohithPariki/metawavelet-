import torch
from config import *
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.meta_wavelet import HermiteBasis

def build_family():
    Jx = torch.arange(0.0, 10.0)
    a = 1
    family = torch.tensor([(2**jx, kx) for jx in Jx for kx in range(int(torch.floor((x_lower-a)*2**(jx))), int(torch.ceil((x_upper+a)*2**(jx))) + 1)])
    return family.to(device)

family = build_family()
N_f = len(family)
print("family_len: ", N_f)

def compute_inputs(pts, family_dim):
    j = family_dim[:, 0]
    k = family_dim[:, 1]
    return j[:, None] * pts[None, :] - k[:, None]

X_coll = compute_inputs(x_collocation, family)
X_bc_left = compute_inputs(x_bc_left, family)
X_bc_right = compute_inputs(x_bc_right, family)
X_val = compute_inputs(x_validation, family)

class WaveletEvaluator:
    def __init__(self, N_H=2):
        self.basis = HermiteBasis(N_H=N_H).to(device)
        self.jx = family[:, 0:1] # [N_f, 1]
        
    def evaluate(self, c, b):
        # c is [N_f]
        psi_X = self.basis.evaluate(X_coll, derivative=0) 
        dpsi_X = self.basis.evaluate(X_coll, derivative=1) 
        d2psi_X = self.basis.evaluate(X_coll, derivative=2) 
        
        W_x = psi_X.T      # [N_coll, N_f]
        DW_x = (self.jx * dpsi_X).T 
        DW2_x = (self.jx**2 * d2psi_X).T 
        
        u = torch.sum(W_x * c, dim=1) + b
        u_x = torch.sum(DW_x * c, dim=1)
        u_xx = torch.sum(DW2_x * c, dim=1)
        
        # BC
        W_bcl = self.basis.evaluate(X_bc_left, derivative=0).T
        u_bcl = torch.sum(W_bcl * c, dim=1) + b
        
        W_bcr = self.basis.evaluate(X_bc_right, derivative=0).T
        u_bcr = torch.sum(W_bcr * c, dim=1) + b
        
        return u, u_x, u_xx, u_bcl, u_bcr
        
    def evaluate_val(self, c, b):
        W_val = self.basis.evaluate(X_val, derivative=0).T
        return torch.sum(W_val * c, dim=1) + b
