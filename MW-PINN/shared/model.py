import torch
import torch.nn as nn
import torch.nn.init as init

class WPINN(nn.Module):
    def __init__(self, input_size, num_hidden_layers1, num_hidden_layers2, hidden_neurons, family_size, spatial_dim=2):
        super(WPINN, self).__init__()
        self.activation = nn.Tanh()
        
        first_stage_layers = []
        first_stage_layers.append(nn.Linear(spatial_dim, hidden_neurons))
        first_stage_layers.append(self.activation)
        
        for _ in range(num_hidden_layers1-1):
            first_stage_layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            first_stage_layers.append(self.activation)
        
        first_stage_layers.append(nn.Linear(hidden_neurons, 1))
        self.first_stage = nn.Sequential(*first_stage_layers)
        
        second_stage_layers = []
        second_stage_layers.append(nn.Linear(input_size, hidden_neurons))
        second_stage_layers.append(self.activation)
        
        for _ in range(num_hidden_layers2-1):
            second_stage_layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            second_stage_layers.append(self.activation)
        
        second_stage_layers.append(nn.Linear(hidden_neurons, family_size))
        self.second_stage = nn.Sequential(*second_stage_layers)
        
        for network in [self.first_stage, self.second_stage]:
            for m in network:
                if isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)
                    init.constant_(m.bias, 0)
        
        self.bias = nn.Parameter(torch.tensor(0.5))

    def forward(self, *coords):
        inputs = torch.stack(coords, dim=-1)
        point_features = self.first_stage(inputs)  
        point_features = point_features.squeeze(-1)  
        coefficients = self.second_stage(point_features)
        return coefficients, self.bias

class CoefficientRefinementNetwork(nn.Module):
    def __init__(self, initial_coefficients, initial_bias):
        super(CoefficientRefinementNetwork, self).__init__()
        self.coefficients = nn.Parameter(initial_coefficients.clone().detach())
        self.bias = nn.Parameter(initial_bias.clone().detach())

    def forward(self, x, t):
        return self.coefficients, self.bias
