import torch
import torch.optim as optim
from tqdm import tqdm
import time
import os

from config import *
from Wfamily import *
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Model import WPINN, CoefficientRefinementNetwork

evaluator = WaveletEvaluator(N_H=2)

wpinn_model = WPINN(input_size=len_collocation, 
                    family_size=len_family,
                    num_hidden_layers1=2, 
                    num_hidden_layers2=3, 
                    hidden_neurons1=50, 
                    hidden_neurons2=50).to(device)

optimizer1 = optim.AdamW(list(wpinn_model.parameters()) + list(evaluator.basis.parameters()), lr=1e-4, weight_decay=1e-4)

def wpinn_loss(coeffs, biases):
    c_u, c_v, c_p = coeffs
    b_u, b_v, b_p = biases
    
    u, u_x, u_y, u_xx, u_yy = evaluator.evaluate_navier_stokes(c_u, b_u, is_u=True)
    v, v_x, v_y, v_xx, v_yy = evaluator.evaluate_navier_stokes(c_v, b_v, is_v=True)
    p, p_x, p_y = evaluator.evaluate_navier_stokes(c_p, b_p, is_p=True)
    
    pde_loss_1 = torch.mean((u * u_x + v * u_y + p_x - (1/Re) * (u_xx + u_yy))**2)
    pde_loss_2 = torch.mean((u * v_x + v * v_y + p_y - (1/Re) * (v_xx + v_yy))**2)
    pde_loss_3 = torch.mean((u_x + v_y)**2)
    pde_loss = pde_loss_1 + pde_loss_2 + pde_loss_3
    
    u_bc_low, u_bc_up, u_bc_left, u_bc_right = evaluator.evaluate_bc(c_u, b_u)
    v_bc_low, v_bc_up, v_bc_left, v_bc_right = evaluator.evaluate_bc(c_v, b_v)
    
    # Boundary Conditions
    bc_loss_u = (torch.mean((u_bc_low - 0)**2) + torch.mean((u_bc_up - 1)**2) + 
                 torch.mean((u_bc_left - 0)**2) + torch.mean((u_bc_right - 0)**2))
    bc_loss_v = (torch.mean((v_bc_low - 0)**2) + torch.mean((v_bc_up - 0)**2) + 
                 torch.mean((v_bc_left - 0)**2) + torch.mean((v_bc_right - 0)**2))
    bc_loss = bc_loss_u + bc_loss_v
    
    total_loss = pde_loss + bc_loss
    return total_loss, pde_loss, bc_loss

def train_phase(model, optimizer, num_epochs, phase_name="Phase"):
    print(f"\n--- Starting {phase_name} ({num_epochs} epochs) ---")
    start_time = time.time()
    
    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        
        coeffs, biases = model(x_collocation, y_collocation)
        total_loss, pde_loss, bc_loss = wpinn_loss(coeffs, biases)
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 5000 == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                c_val, b_val = model(x_collocation, y_collocation)
                u_test = evaluator.evaluate_test(c_val[0], b_val[0])
                v_test = evaluator.evaluate_test(c_val[1], b_val[1])
                pred_vel = torch.sqrt(u_test**2 + v_test**2).reshape(n_test, n_test)
                errL2 = torch.norm(vel_ref - pred_vel) / torch.norm(vel_ref)
                
                tqdm.write(f'Epoch [{epoch}/{num_epochs-1}] Loss: {total_loss.item():.4f} '
                           f'PDE: {pde_loss.item():.4f} BC: {bc_loss.item():.4f} '
                           f'| RelL2: {errL2.item():.4f}')
                           
    print(f"{phase_name} completed in {(time.time() - start_time)/60:.1f} minutes.")
    return coeffs, biases

print("========== META-WAVELET PINN ==========")
print("Problem: Lid-Driven Cavity (Re=100)")
print(f"Device: {device}")
print(f"Collocation points: {len_collocation}")
print(f"Wavelet Family Size: {len_family} (X: {N_x}, Y: {N_y})")
print("=======================================")

coeffs_mid, biases_mid = train_phase(wpinn_model, optimizer1, num_epochs=20001, phase_name="Phase 1 (NN + Shape)")

for param in evaluator.basis.parameters():
    param.requires_grad = False

print("\nFinal Meta-Wavelet Coefficients (a_n):", evaluator.basis.a_n.data.cpu().numpy())

# Phase 2: L-BFGS Coefficient Refinement
with torch.no_grad():
    c_final, b_final = wpinn_model(x_collocation, y_collocation)

refinement_model = CoefficientRefinementNetwork(
    initial_u=c_final[0], initial_v=c_final[1], initial_p=c_final[2],
    b_u=b_final[0], b_v=b_final[1], b_p=b_final[2]
).to(device)

print("\n--- Starting Phase 2 (L-BFGS Coefficient Refinement) ---")
start_time = time.time()
optimizer_lbfgs = optim.LBFGS(refinement_model.parameters(), max_iter=5000, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=50)

def closure():
    optimizer_lbfgs.zero_grad()
    coeffs_bfgs, biases_bfgs = refinement_model(x_collocation, y_collocation)
    loss, pde_loss, bc_loss = wpinn_loss(coeffs_bfgs, biases_bfgs)
    loss.backward()
    return loss

optimizer_lbfgs.step(closure)
print(f"Phase 2 (L-BFGS) completed in {(time.time() - start_time)/60:.1f} minutes.")

with torch.no_grad():
    c_end, b_end = refinement_model(x_collocation, y_collocation)
    u_test = evaluator.evaluate_test(c_end[0], b_end[0])
    v_test = evaluator.evaluate_test(c_end[1], b_end[1])
    pred_vel = torch.sqrt(u_test**2 + v_test**2).reshape(n_test, n_test)
    errL2 = torch.norm(vel_ref - pred_vel) / torch.norm(vel_ref)
    print(f'L-BFGS Final | RelL2: {errL2.item():.6f}')

import matplotlib.pyplot as plt

with torch.no_grad():
    u_test = evaluator.evaluate_test(c_end[0], b_end[0])
    v_test = evaluator.evaluate_test(c_end[1], b_end[1])
    pred_vel = torch.sqrt(u_test**2 + v_test**2).cpu().numpy().reshape(n_test, n_test)

vel_ref_np = vel_ref.cpu().numpy().reshape(n_test, n_test)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
c1 = axes[0].contourf(x_test.cpu().numpy().reshape(n_test, n_test), 
                      y_test.cpu().numpy().reshape(n_test, n_test), vel_ref_np, 100, cmap='jet')
axes[0].set_title('Reference Velocity')
fig.colorbar(c1, ax=axes[0])

c2 = axes[1].contourf(x_test.cpu().numpy().reshape(n_test, n_test), 
                      y_test.cpu().numpy().reshape(n_test, n_test), pred_vel, 100, cmap='jet')
axes[1].set_title('MW-PINN Prediction')
fig.colorbar(c2, ax=axes[1])

c3 = axes[2].contourf(x_test.cpu().numpy().reshape(n_test, n_test), 
                      y_test.cpu().numpy().reshape(n_test, n_test), np.abs(vel_ref_np - pred_vel), 100, cmap='jet')
axes[2].set_title('Absolute Error')
fig.colorbar(c3, ax=axes[2])

os.makedirs('../Results_Comparison', exist_ok=True)
plt.savefig('../Results_Comparison/LidDriven_Re100_sol.png', dpi=150)
print("\nPlot saved to ../Results_Comparison/LidDriven_Re100_sol.png")
