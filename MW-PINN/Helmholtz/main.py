import torch
import torch.optim as optim
from tqdm import tqdm
import time
import os

from config import *
from Helmholtz import *
from Wfamily import *
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.model import WPINN, CoefficientRefinementNetwork

evaluator = WaveletEvaluator(N_H=2)

wpinn_model = WPINN(input_size=n_collocation, 
                    num_hidden_layers1=2, 
                    num_hidden_layers2=4, 
                    hidden_neurons=50, 
                    family_size=len_family).to(device)

optimizer1 = optim.AdamW(list(wpinn_model.parameters()) + list(evaluator.basis.parameters()), lr=1e-4, weight_decay=1e-4)

def wpinn_loss(c, b):
    u, u_xx, u_yy, u_pred_bcl, u_pred_bcr, u_pred_bcb, u_pred_bct = evaluator.evaluate(c, b)
    
    pde_loss = torch.mean((u_xx + u_yy + u - rhs)**2)
    bc_loss = (torch.mean((u_pred_bcl - u_bc_left)**2) + 
               torch.mean((u_pred_bcr - u_bc_right)**2) +
               torch.mean((u_pred_bcb - u_bc_bottom)**2) + 
               torch.mean((u_pred_bct - u_bc_top)**2))
    
    total_loss = pde_loss + bc_loss
    return total_loss, pde_loss, bc_loss

def train_phase(model, optimizer, num_epochs, phase_name="Phase"):
    print(f"\n--- Starting {phase_name} ({num_epochs} epochs) ---")
    start_time = time.time()
    
    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        
        c, b = model(x_collocation, y_collocation)
        total_loss, pde_loss, bc_loss = wpinn_loss(c, b)
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 4000 == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                c_val, b_val = model(x_collocation, y_collocation)
                num_val = evaluator.evaluate_val(c_val, b_val)
                errL2 = torch.norm(exact_validation - num_val) / torch.norm(exact_validation)
                errMax = torch.max(torch.abs(exact_validation - num_val))
                
                tqdm.write(f'Epoch [{epoch}/{num_epochs-1}] Loss: {total_loss.item():.4f} '
                           f'PDE: {pde_loss.item():.4f} BC: {bc_loss.item():.4f} '
                           f'| RelL2: {errL2.item():.4f} Max: {errMax.item():.4f}')
                           
    print(f"{phase_name} completed in {(time.time() - start_time)/60:.1f} minutes.")
    return c, b

print("========== META-WAVELET PINN ==========")
print("Problem: Helmholtz")
print(f"Device: {device}")
print(f"Collocation points: {n_collocation}")
print(f"Wavelet Family Size: {len_family} (X: {N_x}, Y: {N_y})")
print("=======================================")

c_mid, b_mid = train_phase(wpinn_model, optimizer1, num_epochs=20001, phase_name="Phase 1 (NN + Shape)")

for param in evaluator.basis.parameters():
    param.requires_grad = False

print("\nFinal Meta-Wavelet Coefficients (a_n):", evaluator.basis.a_n.data.cpu().numpy())

# Phase 2: L-BFGS Coefficient Refinement
with torch.no_grad():
    c_final, b_final = wpinn_model(x_collocation, y_collocation)

refinement_model = CoefficientRefinementNetwork(initial_coefficients=c_final, initial_bias=b_final).to(device)

print("\n--- Starting Phase 2 (L-BFGS Coefficient Refinement) ---")
start_time = time.time()
optimizer_lbfgs = optim.LBFGS(refinement_model.parameters(), max_iter=5000, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=50)

def closure():
    optimizer_lbfgs.zero_grad()
    c_bfgs, b_bfgs = refinement_model(x_collocation, y_collocation)
    loss, pde_loss, bc_loss = wpinn_loss(c_bfgs, b_bfgs)
    loss.backward()
    return loss

optimizer_lbfgs.step(closure)
print(f"Phase 2 (L-BFGS) completed in {(time.time() - start_time)/60:.1f} minutes.")

with torch.no_grad():
    c_end, b_end = refinement_model(x_collocation, y_collocation)
    num_val = evaluator.evaluate_val(c_end, b_end)
    errL2 = torch.norm(exact_validation - num_val) / torch.norm(exact_validation)
    errMax = torch.max(torch.abs(exact_validation - num_val))
    print(f'L-BFGS Final | RelL2: {errL2.item():.6f} Max: {errMax.item():.6f}')

import matplotlib.pyplot as plt

xtest_np = x_test.cpu().numpy().reshape(n_test, n_test)
ytest_np = y_test.cpu().numpy().reshape(n_test, n_test)

with torch.no_grad():
    u_test_pred = evaluator.evaluate_test(c_end, b_end).cpu().numpy().reshape(n_test, n_test)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
c1 = axes[0].contourf(xtest_np, ytest_np, exact_test, 100, cmap='jet')
axes[0].set_title('Exact Solution')
fig.colorbar(c1, ax=axes[0])

c2 = axes[1].contourf(xtest_np, ytest_np, u_test_pred, 100, cmap='jet')
axes[1].set_title('MW-PINN Prediction')
fig.colorbar(c2, ax=axes[1])

c3 = axes[2].contourf(xtest_np, ytest_np, np.abs(exact_test - u_test_pred), 100, cmap='jet')
axes[2].set_title('Absolute Error')
fig.colorbar(c3, ax=axes[2])

os.makedirs('../Results_Comparison', exist_ok=True)
plt.savefig('../Results_Comparison/Helmholtz_sol.png', dpi=150)
print("\nPlot saved to ../Results_Comparison/Helmholtz_sol.png")
