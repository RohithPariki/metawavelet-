import torch
from config import *
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.meta_wavelet import HermiteBasis

def build_family():
    Jx = torch.arange(-5.0, 6.0)
    Jt = torch.arange(-5.0, 6.0)
    a = 0.5
    family_x = torch.tensor([(2**jx, kx) for jx in Jx for kx in range(int(torch.floor((x_lower-a)*2**(jx))), int(torch.ceil((x_upper+a)*2**(jx))) + 1)])
    family_t = torch.tensor([(2**jt, kt) for jt in Jt for kt in range(int(torch.floor((t_lower-a)*2**(jt))), int(torch.ceil((t_upper+a)*2**(jt))) + 1)])
    return family_x.to(device), family_t.to(device)

family_x, family_t = build_family()
N_x = len(family_x)
N_t = len(family_t)
len_family = N_x * N_t
print("family_len: ", len_family)

def compute_inputs(pts, family_dim):
    j = family_dim[:, 0]
    k = family_dim[:, 1]
    return j[:, None] * pts[None, :] - k[:, None]

# Subdomain 1
X_coll1 = compute_inputs(x_collocation1, family_x)
# Subdomain 2
X_coll2 = compute_inputs(x_collocation2, family_x)
T_coll = compute_inputs(t_collocation, family_t)

# IC
X_ic1 = compute_inputs(x_ic1, family_x)
X_ic2 = compute_inputs(x_ic2, family_x)
T_ic = compute_inputs(t_ic, family_t)

# BC
X_bc_left = compute_inputs(x_bc_left, family_x)
X_bc_right = compute_inputs(x_bc_right, family_x)
T_bc = compute_inputs(t_bc, family_t)

# Interface
X_int = compute_inputs(x_interface, family_x)
T_int = compute_inputs(t_interface, family_t)

# Val / Test
X_val1 = compute_inputs(x_validation1, family_x)
X_val2 = compute_inputs(x_validation2, family_x)
T_val = compute_inputs(t_validation, family_t)

X_test1 = compute_inputs(x_test1.to(device), family_x)
X_test2 = compute_inputs(x_test2.to(device), family_x)
T_test = compute_inputs(t_test.to(device), family_t)

class WaveletEvaluator:
    def __init__(self, N_H=2):
        self.basis = HermiteBasis(N_H=N_H).to(device)
        self.jx = family_x[:, 0:1] # [N_x, 1]
        self.jt = family_t[:, 0:1] # [N_t, 1]
        
    def evaluate_collocation(self, c_flat, b, is_subdomain2=False):
        c = c_flat.view(N_x, N_t)
        
        X = X_coll2 if is_subdomain2 else X_coll1
        
        psi_X = self.basis.evaluate(X, derivative=0) 
        psi_T = self.basis.evaluate(T_coll, derivative=0)
        dpsi_X = self.basis.evaluate(X, derivative=1)
        dpsi_T = self.basis.evaluate(T_coll, derivative=1)
        
        W_x = psi_X.T
        W_t = psi_T.T
        DW_x = (self.jx * dpsi_X).T
        DW_t = (self.jt * dpsi_T).T
        
        u = torch.sum((W_x @ c) * W_t, dim=1) + b
        u_x = torch.sum((DW_x @ c) * W_t, dim=1)
        u_t = torch.sum((W_x @ c) * DW_t, dim=1)
        
        return u, u_x, u_t
        
    def evaluate_ic(self, c_flat, b, is_subdomain2=False):
        c = c_flat.view(N_x, N_t)
        X = X_ic2 if is_subdomain2 else X_ic1
        
        W_x = self.basis.evaluate(X, derivative=0).T
        W_t = self.basis.evaluate(T_ic, derivative=0).T
        return torch.sum((W_x @ c) * W_t, dim=1) + b
        
    def evaluate_bc(self, c_flat, b, is_right=False):
        c = c_flat.view(N_x, N_t)
        X = X_bc_right if is_right else X_bc_left
        
        W_x = self.basis.evaluate(X, derivative=0).T
        W_t = self.basis.evaluate(T_bc, derivative=0).T
        return torch.sum((W_x @ c) * W_t, dim=1) + b
        
    def evaluate_interface(self, c_flat, b):
        c = c_flat.view(N_x, N_t)
        W_x = self.basis.evaluate(X_int, derivative=0).T
        W_t = self.basis.evaluate(T_int, derivative=0).T
        return torch.sum((W_x @ c) * W_t, dim=1) + b
        
    def evaluate_val(self, c_flat, b, is_subdomain2=False):
        c = c_flat.view(N_x, N_t)
        X = X_val2 if is_subdomain2 else X_val1
        W_x = self.basis.evaluate(X, derivative=0).T
        W_t = self.basis.evaluate(T_val, derivative=0).T
        return torch.sum((W_x @ c) * W_t, dim=1) + b
        
    def evaluate_test(self, c_flat, b, is_subdomain2=False):
        c = c_flat.view(N_x, N_t)
        X = X_test2 if is_subdomain2 else X_test1
        W_x = self.basis.evaluate(X, derivative=0).T
        W_t = self.basis.evaluate(T_test, derivative=0).T
        return torch.sum((W_x @ c) * W_t, dim=1) + b
