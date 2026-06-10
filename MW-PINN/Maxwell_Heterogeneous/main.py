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

global mu2, epsilon2
mu2 = 4.5
epsilon2 = 0.5

# Analytical solutions for validation/testing
def analytical1(x, t):
    arg1 = 2*t - 2*x + 1
    arg2 = 2*t + 2*x - 1
    c1 = torch.cos(arg1)
    c2 = torch.cos(arg2)
    E = c1 + 0.5*c2
    H = c1 - 0.5*c2
    return E, H

def analytical2(x, t):
    arg3 = 2*t - 3*x + 1.5
    c3 = torch.cos(arg3)
    E = 1.5*c3
    H = 0.5*c3
    return E, H

E_validation1, H_validation1 = analytical1(x_validation1, t_validation)
E_validation2, H_validation2 = analytical2(x_validation2, t_validation)
E_validation = torch.cat((E_validation1, E_validation2))
H_validation = torch.cat((H_validation1, H_validation2))

E_ic1, H_ic1 = analytical1(x_ic1, t_ic)
E_ic2, H_ic2 = analytical2(x_ic2, t_ic)
E_bc_left, H_bc_left = analytical1(x_bc_left, t_bc)
E_bc_right, H_bc_right = analytical2(x_bc_right, t_bc)

E_exact1, H_exact1 = analytical1(x_test1.to(device), t_test.to(device))
E_exact2, H_exact2 = analytical2(x_test2.to(device), t_test.to(device))
E_exact = torch.cat((E_exact1, E_exact2)).reshape(n_test, n_test).cpu().numpy()
H_exact = torch.cat((H_exact1, H_exact2)).reshape(n_test, n_test).cpu().numpy()

evaluator = WaveletEvaluator(N_H=2)

wpinn_model1 = WPINN(input_size=n_collocation, num_hidden_layers1=2, num_hidden_layers2=4, hidden_neurons=50, family_size=len_family).to(device)
wpinn_model2 = WPINN(input_size=n_collocation, num_hidden_layers1=2, num_hidden_layers2=4, hidden_neurons=50, family_size=len_family).to(device)

optimizer1 = optim.Adam(list(wpinn_model1.parameters()) + list(wpinn_model2.parameters()) + list(evaluator.basis.parameters()), lr=1e-4)

def wpinn_loss(c1, b1, c2, b2):
    c_E1, c_H1 = c1
    b_E1, b_H1 = b1
    c_E2, c_H2 = c2
    b_E2, b_H2 = b2
    
    # Subdomain 1
    E1, E_x1, E_t1 = evaluator.evaluate_collocation(c_E1, b_E1, is_subdomain2=False)
    H1, H_x1, H_t1 = evaluator.evaluate_collocation(c_H1, b_H1, is_subdomain2=False)
    
    # Subdomain 2
    E2, E_x2, E_t2 = evaluator.evaluate_collocation(c_E2, b_E2, is_subdomain2=True)
    H2, H_x2, H_t2 = evaluator.evaluate_collocation(c_H2, b_H2, is_subdomain2=True)
    
    pde_loss = torch.mean(torch.cat((E_x1+H_t1, E_x2+mu2*H_t2))**2) + \
               torch.mean(torch.cat((H_x1+E_t1, H_x2+epsilon2*E_t2))**2)
               
    # IC
    E_pred_ic1 = evaluator.evaluate_ic(c_E1, b_E1, is_subdomain2=False)
    H_pred_ic1 = evaluator.evaluate_ic(c_H1, b_H1, is_subdomain2=False)
    E_pred_ic2 = evaluator.evaluate_ic(c_E2, b_E2, is_subdomain2=True)
    H_pred_ic2 = evaluator.evaluate_ic(c_H2, b_H2, is_subdomain2=True)
    ic_loss = torch.mean((E_pred_ic1 - E_ic1)**2) + torch.mean((H_pred_ic1 - H_ic1)**2) + \
              torch.mean((E_pred_ic2 - E_ic2)**2) + torch.mean((H_pred_ic2 - H_ic2)**2)
              
    # BC
    E_pred_bcl = evaluator.evaluate_bc(c_E1, b_E1, is_right=False)
    H_pred_bcl = evaluator.evaluate_bc(c_H1, b_H1, is_right=False)
    E_pred_bcr = evaluator.evaluate_bc(c_E2, b_E2, is_right=True)
    H_pred_bcr = evaluator.evaluate_bc(c_H2, b_H2, is_right=True)
    bc_loss = torch.mean((E_pred_bcl - E_bc_left)**2) + torch.mean((H_pred_bcl - H_bc_left)**2) + \
              torch.mean((E_pred_bcr - E_bc_right)**2) + torch.mean((H_pred_bcr - H_bc_right)**2)
              
    # Interface
    E_int1 = evaluator.evaluate_interface(c_E1, b_E1)
    H_int1 = evaluator.evaluate_interface(c_H1, b_H1)
    E_int2 = evaluator.evaluate_interface(c_E2, b_E2)
    H_int2 = evaluator.evaluate_interface(c_H2, b_H2)
    int_loss = torch.mean((E_int1 - E_int2)**2) + torch.mean((H_int1 - H_int2)**2)
    
    total_loss = pde_loss + ic_loss + bc_loss + int_loss
    return total_loss, pde_loss, ic_loss, bc_loss, int_loss

def train_phase(model1, model2, optimizer, num_epochs, phase_name="Phase"):
    print(f"\n--- Starting {phase_name} ({num_epochs} epochs) ---")
    start_time = time.time()
    
    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        
        c1, b1 = model1(x_collocation1, t_collocation)
        c2, b2 = model2(x_collocation2, t_collocation)
        
        total_loss, pde_loss, ic_loss, bc_loss, int_loss = wpinn_loss(c1, b1, c2, b2)
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 5000 == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                c1_val, b1_val = model1(x_collocation1, t_collocation)
                c2_val, b2_val = model2(x_collocation2, t_collocation)
                
                E_val1 = evaluator.evaluate_val(c1_val[0], b1_val[0], is_subdomain2=False)
                E_val2 = evaluator.evaluate_val(c2_val[0], b2_val[0], is_subdomain2=True)
                E_val = torch.cat((E_val1, E_val2))
                
                errL2 = torch.norm(E_validation - E_val) / torch.norm(E_validation)
                
                tqdm.write(f'Epoch [{epoch}/{num_epochs-1}] Loss: {total_loss.item():.4f} '
                           f'PDE: {pde_loss.item():.4f} IC: {ic_loss.item():.4f} BC: {bc_loss.item():.4f} Int: {int_loss.item():.4f} '
                           f'| RelL2: {errL2.item():.4f}')
                           
    print(f"{phase_name} completed in {(time.time() - start_time)/60:.1f} minutes.")
    return c1, b1, c2, b2

print("========== META-WAVELET PINN ==========")
print("Problem: Maxwell's Equation (Heterogeneous)")
print(f"Device: {device}")
print(f"Collocation points: {n_collocation} per subdomain")
print(f"Wavelet Family Size: {len_family} (X: {N_x}, T: {N_t})")
print("=======================================")

c1_mid, b1_mid, c2_mid, b2_mid = train_phase(wpinn_model1, wpinn_model2, optimizer1, num_epochs=30001, phase_name="Phase 1 (NN + Shape)")

for param in evaluator.basis.parameters():
    param.requires_grad = False

print("\nFinal Meta-Wavelet Coefficients (a_n):", evaluator.basis.a_n.data.cpu().numpy())

with torch.no_grad():
    c1_final, b1_final = wpinn_model1(x_collocation1, t_collocation)
    c2_final, b2_final = wpinn_model2(x_collocation2, t_collocation)

ref_model1 = CoefficientRefinementNetwork(initial_coefficients=c1_final, initial_bias=b1_final).to(device)
ref_model2 = CoefficientRefinementNetwork(initial_coefficients=c2_final, initial_bias=b2_final).to(device)

optimizer2 = optim.Adam(list(ref_model1.parameters()) + list(ref_model2.parameters()), lr=1e-3)

c1_end, b1_end, c2_end, b2_end = train_phase(ref_model1, ref_model2, optimizer2, num_epochs=10001, phase_name="Phase 2 (Coeff Refinement)")

import matplotlib.pyplot as plt

with torch.no_grad():
    E_test1 = evaluator.evaluate_test(c1_end[0], b1_end[0], is_subdomain2=False)
    E_test2 = evaluator.evaluate_test(c2_end[0], b2_end[0], is_subdomain2=True)
    E_test_pred = torch.cat((E_test1, E_test2)).cpu().numpy().reshape(n_test, n_test)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# We need to compute the grid correctly for plotting
xtest1_np = xtest1.cpu().numpy()
xtest2_np = xtest2.cpu().numpy()
xtest_np = np.concatenate((xtest1_np, xtest2_np))
ttest_np = ttest.cpu().numpy()

T_grid, X_grid = np.meshgrid(ttest_np, xtest_np)

c1 = axes[0].contourf(T_grid, X_grid, E_exact, 100, cmap='jet')
axes[0].set_title('Exact Solution (E)')
fig.colorbar(c1, ax=axes[0])

c2 = axes[1].contourf(T_grid, X_grid, E_test_pred, 100, cmap='jet')
axes[1].set_title('MW-PINN Prediction (E)')
fig.colorbar(c2, ax=axes[1])

c3 = axes[2].contourf(T_grid, X_grid, np.abs(E_exact - E_test_pred), 100, cmap='jet')
axes[2].set_title('Absolute Error')
fig.colorbar(c3, ax=axes[2])

os.makedirs('../Results_Comparison', exist_ok=True)
plt.savefig('../Results_Comparison/Maxwell_Hetero_sol.png', dpi=150)
print("\nPlot saved to ../Results_Comparison/Maxwell_Hetero_sol.png")
