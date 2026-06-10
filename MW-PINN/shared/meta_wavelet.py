import torch
import torch.nn as nn

class HermiteBasis(nn.Module):
    def __init__(self, N_H=4):
        super().__init__()
        self.N_H = N_H
        init_a = torch.zeros(N_H)
        if N_H >= 1:
            init_a[0] = 0.5
        self.a_n = nn.Parameter(init_a)
        
    def hermite_functions(self, u, max_degree):
        phi = []
        ex = torch.exp(-(u**2)/2)
        phi.append(ex)
        if max_degree == 0: return phi
        phi.append(2 * u * ex)
        if max_degree == 1: return phi
            
        for n in range(1, max_degree):
            phi_next = 2 * u * phi[n] - 2 * n * phi[n-1]
            phi.append(phi_next)
        return phi

    def evaluate(self, u, derivative=0):
        max_deg = self.N_H + derivative
        phi = self.hermite_functions(u, max_deg)
        res = torch.zeros_like(u)
        sign = (-1)**derivative
        for n in range(1, self.N_H + 1):
            res += self.a_n[n-1] * phi[n + derivative]
        return sign * res
