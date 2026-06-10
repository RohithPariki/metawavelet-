import torch
import torch.optim as optim
from tqdm import tqdm
import time
import os

from config import *
from SPP_problem import *
from Wfamily import *
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.model import WPINN, CoefficientRefinementNetwork

evaluator = WaveletEvaluator(N_H=2)

wpinn_model = WPINN(input_size=n_collocation, 
                    num_hidden_layers1=2, 
                    num_hidden_layers2=4, 
                    hidden_neurons=50, 
                    family_size=N_f,
                    spatial_dim=1).to(device)

optimizer1 = optim.AdamW(list(wpinn_model.parameters()) + list(evaluator.basis.parameters()), lr=1e-4, weight_decay=1e-4)

def wpinn_loss(c, b):
    u, u_x, u_xx, u_pred_bcl, u_pred_bcr = evaluator.evaluate(c, b)
    
    pde_loss = torch.mean((e*u_xx + u_x + u)**2)
    bc_loss = torch.mean((u_pred_bcl - u_bc_left)**2) + torch.mean((u_pred_bcr - u_bc_right)**2)
    
    total_loss = pde_loss + bc_loss
    return total_loss, pde_loss, bc_loss

def train_phase(model, optimizer, num_epochs, phase_name="Phase"):
    print(f"\n--- Starting {phase_name} ({num_epochs} epochs) ---")
    start_time = time.time()
    
    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        
        c, b = model(x_collocation)
        total_loss, pde_loss, bc_loss = wpinn_loss(c, b)
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 4000 == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                c_val, b_val = model(x_collocation)
                num_val = evaluator.evaluate_val(c_val, b_val)
                errL2 = torch.norm(exact - num_val) / torch.norm(exact)
                errMax = torch.max(torch.abs(exact - num_val))
                
                tqdm.write(f'Epoch [{epoch}/{num_epochs-1}] Loss: {total_loss.item():.4f} '
                           f'PDE: {pde_loss.item():.4f} BC: {bc_loss.item():.4f} '
                           f'| RelL2: {errL2.item():.4f} Max: {errMax.item():.4f}')
                           
    print(f"{phase_name} completed in {(time.time() - start_time)/60:.1f} minutes.")
    return c, b

print("========== META-WAVELET PINN ==========")
print("Problem: SPP Example 1")
print(f"Device: {device}")
print(f"Collocation points: {n_collocation}")
print(f"Wavelet Family Size: {N_f}")
print("=======================================")

c_mid, b_mid = train_phase(wpinn_model, optimizer1, num_epochs=20001, phase_name="Phase 1 (NN + Shape)")

for param in evaluator.basis.parameters():
    param.requires_grad = False

print("\nFinal Meta-Wavelet Coefficients (a_n):", evaluator.basis.a_n.data.cpu().numpy())

# Phase 2: L-BFGS Coefficient Refinement
with torch.no_grad():
    c_final, b_final = wpinn_model(x_collocation)

refinement_model = CoefficientRefinementNetwork(initial_coefficients=c_final, initial_bias=b_final).to(device)

print("\n--- Starting Phase 2 (L-BFGS Coefficient Refinement) ---")
start_time = time.time()
optimizer_lbfgs = optim.LBFGS(refinement_model.parameters(), max_iter=5000, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=50)

def closure():
    optimizer_lbfgs.zero_grad()
    c_bfgs, b_bfgs = refinement_model(x_collocation)
    loss, pde_loss, bc_loss = wpinn_loss(c_bfgs, b_bfgs)
    loss.backward()
    return loss

optimizer_lbfgs.step(closure)
print(f"Phase 2 (L-BFGS) completed in {(time.time() - start_time)/60:.1f} minutes.")

with torch.no_grad():
    c_end, b_end = refinement_model(x_collocation)
    num_val = evaluator.evaluate_val(c_end, b_end)
    errL2 = torch.norm(exact - num_val) / torch.norm(exact)
    errMax = torch.max(torch.abs(exact - num_val))
    print(f'L-BFGS Final | RelL2: {errL2.item():.6f} Max: {errMax.item():.6f}')

import matplotlib.pyplot as plt

with torch.no_grad():
    u_test_pred = evaluator.evaluate_val(c_end, b_end).cpu().numpy()

plt.figure(figsize=(8, 6))
plt.plot(x_test.cpu().numpy(), u_exact, 'b-', label='Exact', linewidth=2)
plt.plot(x_validation.cpu().numpy(), u_test_pred, 'r--', label='MW-PINN')
plt.title('SPP Example 1')
plt.legend()
plt.grid(True)

os.makedirs('../Results_Comparison', exist_ok=True)
plt.savefig('../Results_Comparison/SPP_Ex1_sol.png', dpi=150)
print("\nPlot saved to ../Results_Comparison/SPP_Ex1_sol.png")
