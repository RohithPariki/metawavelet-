import torch
import torch.nn as nn
import torch.nn.init as init

class WPINN(nn.Module):
    def __init__(self, input_size, family_size, num_hidden_layers1, num_hidden_layers2, hidden_neurons1, hidden_neurons2):
        super(WPINN, self).__init__()
        self.activation = nn.Tanh()
        
        first_stage_layers = []
        first_stage_layers.append(nn.Linear(2, hidden_neurons1))
        first_stage_layers.append(self.activation)
        
        for _ in range(num_hidden_layers1):
            first_stage_layers.append(nn.Linear(hidden_neurons1, hidden_neurons1))
            first_stage_layers.append(self.activation)
        
        first_stage_layers.append(nn.Linear(hidden_neurons1, 1))
        self.first_stage = nn.Sequential(*first_stage_layers)
        
        self.second_stage_u = self.create_second_stage(input_size, family_size, num_hidden_layers2, hidden_neurons2)
        self.second_stage_v = self.create_second_stage(input_size, family_size, num_hidden_layers2, hidden_neurons2)
        self.second_stage_p = self.create_second_stage(input_size, family_size, num_hidden_layers2, hidden_neurons2)
        
        for network in [self.first_stage, self.second_stage_u, self.second_stage_v, self.second_stage_p]:
            for m in network:
                if isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)
                    init.constant_(m.bias, 0)

        self.bias_u = nn.Parameter(torch.tensor(0.5))
        self.bias_v = nn.Parameter(torch.tensor(0.5))
        self.bias_p = nn.Parameter(torch.tensor(0.5))
        
    def create_second_stage(self, input_size, family_size, num_layers, hidden_neurons):
        layers = []
        layers.append(nn.Linear(input_size, hidden_neurons))
        layers.append(self.activation)
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(self.activation)
        layers.append(nn.Linear(hidden_neurons, family_size))
        return nn.Sequential(*layers)

    def forward(self, x, y):
        inputs = torch.stack([x, y], dim=-1)
        point_features = self.first_stage(inputs)
        point_features = point_features.squeeze(-1)

        coeff_u = self.second_stage_u(point_features)
        coeff_v = self.second_stage_v(point_features)
        coeff_p = self.second_stage_p(point_features)
        
        return (coeff_u, coeff_v, coeff_p), (self.bias_u, self.bias_v, self.bias_p)

class CoefficientRefinementNetwork(nn.Module):
    def __init__(self, initial_u, initial_v, initial_p, b_u, b_v, b_p):
        super(CoefficientRefinementNetwork, self).__init__()
        self.c_u = nn.Parameter(initial_u.clone().detach())
        self.c_v = nn.Parameter(initial_v.clone().detach())
        self.c_p = nn.Parameter(initial_p.clone().detach())
        self.b_u = nn.Parameter(b_u.clone().detach())
        self.b_v = nn.Parameter(b_v.clone().detach())
        self.b_p = nn.Parameter(b_p.clone().detach())

    def forward(self, x, y):
        return (self.c_u, self.c_v, self.c_p), (self.b_u, self.b_v, self.b_p)
