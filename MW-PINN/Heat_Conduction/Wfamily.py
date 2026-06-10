import torch
from config import *
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.meta_wavelet import HermiteBasis

def build_family():
    Jx = torch.arange(-6.0, 6.0)
    Jt = torch.arange(-6.0, 6.0)
    a = 0.2
    family_x = torch.tensor([(2**jx, kx) for jx in Jx for kx in range(int(torch.floor((x_lower-a)*2**(jx))), int(torch.ceil((x_upper+a)*2**(jx))) + 1)])
    family_t = torch.tensor([(2**jt, kt) for jt in Jt for kt in range(int(torch.floor((t_lower-a)*2**(jt))), int(torch.ceil((t_upper+a)*2**(jt))) + 1)])
    return family_x.to(device), family_t.to(device)

family_x, family_t = build_family()
N_x = len(family_x)
N_t = len(family_t)
len_family = N_x * N_t
print("family_len: ", len_family)

def compute_inputs(pts, family_dim):
    # pts: [N_pts]
    # family_dim: [N_f, 2] where [:,0]=j, [:,1]=k
    j = family_dim[:, 0]
    k = family_dim[:, 1]
    return j[:, None] * pts[None, :] - k[:, None]  # [N_f, N_pts]

# Collocation inputs
X_coll = compute_inputs(x_collocation, family_x)  # [N_x, N_coll]
T_coll = compute_inputs(t_collocation, family_t)  # [N_t, N_coll]

# Boundary / Initial inputs
X_ic = compute_inputs(x_ic, family_x)
T_ic = compute_inputs(t_ic, family_t)
X_bc_left = compute_inputs(x_bc_left, family_x)
X_bc_right = compute_inputs(x_bc_right, family_x)
T_bc = compute_inputs(t_bc, family_t)

# Validation inputs
X_val = compute_inputs(x_validation, family_x)
T_val = compute_inputs(t_validation, family_t)

# Test inputs
X_test = compute_inputs(x_test.to(device), family_x)
T_test = compute_inputs(t_test.to(device), family_t)

class WaveletEvaluator:
    def __init__(self, N_H=2):
        self.basis = HermiteBasis(N_H=N_H).to(device)
        self.jt = family_t[:, 0:1] # [N_t, 1]
        self.jx = family_x[:, 0:1] # [N_x, 1]
        
    def evaluate(self, c_flat, b):
        c = c_flat.view(N_x, N_t)
        
        # Collocation
        psi_X = self.basis.evaluate(X_coll, derivative=0)  # [N_x, N_coll]
        psi_T = self.basis.evaluate(T_coll, derivative=0)  # [N_t, N_coll]
        dpsi_T = self.basis.evaluate(T_coll, derivative=1) # [N_t, N_coll]
        d2psi_X = self.basis.evaluate(X_coll, derivative=2) # [N_x, N_coll]
        
        W_x = psi_X.T      # [N_coll, N_x]
        W_t = psi_T.T      # [N_coll, N_t]
        DW_t = (self.jt * dpsi_T).T  # [N_coll, N_t]
        DW2_x = (self.jx**2 * d2psi_X).T # [N_coll, N_x]
        
        # Fast batched Einsum evaluation!
        # u = sum_x sum_t W_x[i,x] c[x,t] W_t[i,t]
        u = torch.sum((W_x @ c) * W_t, dim=1) + b
        u_t = torch.sum((W_x @ c) * DW_t, dim=1)
        u_xx = torch.sum((DW2_x @ c) * W_t, dim=1)
        
        # IC and BC
        W_ic_x = self.basis.evaluate(X_ic, derivative=0).T
        W_ic_t = self.basis.evaluate(T_ic, derivative=0).T
        u_ic = torch.sum((W_ic_x @ c) * W_ic_t, dim=1) + b
        
        W_bcl_x = self.basis.evaluate(X_bc_left, derivative=0).T
        W_bc_t = self.basis.evaluate(T_bc, derivative=0).T
        u_bcl = torch.sum((W_bcl_x @ c) * W_bc_t, dim=1) + b
        
        W_bcr_x = self.basis.evaluate(X_bc_right, derivative=0).T
        u_bcr = torch.sum((W_bcr_x @ c) * W_bc_t, dim=1) + b
        
        return u, u_t, u_xx, u_ic, u_bcl, u_bcr
        
    def evaluate_val(self, c_flat, b):
        c = c_flat.view(N_x, N_t)
        W_x = self.basis.evaluate(X_val, derivative=0).T
        W_t = self.basis.evaluate(T_val, derivative=0).T
        return torch.sum((W_x @ c) * W_t, dim=1) + b

    def evaluate_test(self, c_flat, b):
        c = c_flat.view(N_x, N_t)
        W_x = self.basis.evaluate(X_test, derivative=0).T
        W_t = self.basis.evaluate(T_test, derivative=0).T
        return torch.sum((W_x @ c) * W_t, dim=1) + b
