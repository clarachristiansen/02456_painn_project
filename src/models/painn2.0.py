import torch
from torch import nn
from torch_geometric.nn import radius_graph

def rbf_generator(distance, num_rbf_features, cutoff_distance):
    n = torch.arange(num_rbf_features, device=distance.device) + 1
    return torch.sin(distance.unsqueeze(-1) * n * torch.pi / cutoff_distance) / distance.unsqueeze(-1)

def f_cut(distance, cutoff_distance):
    # https://schnetpack.readthedocs.io/en/latest/_modules/schnetpack/nn/cutoff.html#CosineCutoff
    return torch.where(
        distance < cutoff_distance,
        0.5 * (torch.cos(torch.pi * distance / cutoff_distance) + 1),
        torch.tensor(0.0, device=distance.device, dtype=distance.dtype),
    )

class MessageLayer(nn.Module):
    def __init__(self, num_features, num_rbf_features, cutoff_distance):
        super().__init__()
        
        self.num_rbf_features = num_rbf_features
        self.num_features = num_features
        self.cutoff_distance = cutoff_distance
        
        self.phi_path = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features * 3),
        )
        
        self.W_path = nn.Linear(num_rbf_features, num_features * 3)
        
    def forward(self, s, v, edge_indexes, r_ij, distance):
        W = self.W_path(rbf_generator(distance, self.num_rbf_features, self.cutoff_distance))
        W = W * f_cut(distance, self.cutoff_distance).unsqueeze(-1)
        phi = self.phi_path(s)        
        split = W * phi[edge_indexes[:, 1]]
        
        Wvv, Wvs, Wvs = torch.split(
            split, 
            self.num_features,
            dim = 1,
        )
        
        v_1 =  v[edge_indexes[:, 1]] * Wvv.unsqueeze(1) 
        v_2 = Wvs.unsqueeze(1) * (r_ij / distance.unsqueeze(-1)).unsqueeze(-1)
        v_sum = v_1 + v_2
        
        delta_s = torch.zeros_like(s)
        delta_v = torch.zeros_like(v)
        delta_s.index_add_(0, edge_indexes[:, 0], Wvs)
        delta_v.index_add_(0, edge_indexes[:, 0], v_sum)
        
        new_s = s + delta_s
        new_v = v + delta_v
        
        return new_s, new_v

class UpdateLayer(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        
        self.U = nn.Linear(num_features, num_features)
        self.V = nn.Linear(num_features, num_features)
        
        self.mlp_update = nn.Sequential(
            nn.Linear(num_features * 2, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features * 3),
        )
        
    def forward(self, s, v):
        Uv = self.U(v)
        Vv = self.V(v)
        
        Vv_norm = torch.linalg.norm(Vv, dim=1)
        mlp_input = torch.cat((Vv_norm, s), dim=1)
        mlp_result = self.mlp_update(mlp_input)
        
        a_vv, a_sv, a_ss = torch.split(
            mlp_result,                                        
            v.shape[-1],                                       
            dim = 1,
        )
        
        delta_v = a_vv.unsqueeze(1) * Uv
        dot_prod = torch.sum(Uv * Vv, dim=1)
        delta_s = a_sv * dot_prod + a_ss
        
        new_s = s + delta_s
        new_v = v + delta_v
        return new_s, new_v

class PaiNN(nn.Module):
    def __init__(
        self, 
        num_message_passing_layers: int=3, 
        num_features: int = 128, 
        num_outputs: int = 1,
        num_rbf_features: int = 20,
        num_unique_atoms: int = 100,
        cutoff_dist: float = 5.0,
        device: str='cpu'
    ):
        super().__init__()
        
        self.num_unique_atoms = num_unique_atoms   # number of all elements
        self.cutoff_distance = cutoff_dist
        self.num_message_passing_layers = num_message_passing_layers
        self.num_features = num_features
        self.num_rbf_features = num_rbf_features
        self.num_outputs = num_outputs
        
        self.embedding = nn.Embedding(self.num_unique_atoms, self.num_features)

        self.message_layers = nn.ModuleList(
            [
                MessageLayer(self.num_features, self.num_rbf_features, self.cutoff_distance)
                for _ in range(self.num_message_passing_layers)
            ]
        )
        self.update_layers = nn.ModuleList(
            [
                UpdateLayer(self.num_features)
                for _ in range(self.num_message_passing_layers)
            ]            
        )
        
        self.last_mlp = nn.Sequential(
            nn.Linear(self.num_features, self.num_features),
            nn.SiLU(),
            nn.Linear(self.num_features, self.num_outputs),
        )
        
    def forward(self, atoms, atom_positions, graph_indexes):
        edge = build_edge_index(atom_positions, self.cutoff_distance, graph_indexes).T
       
        r_ij = atom_positions[edge[:,1]] - atom_positions[edge[:,0]]

        distance = torch.linalg.norm(r_ij, dim=1)
        
        s = self.embedding(atoms)
        v = torch.zeros((atom_positions.shape[0], 3, self.num_features),
                                  device=r_ij.device,
                                  dtype=r_ij.dtype,
                                 )
        
        for message_layer, update_layer in zip(self.message_layers, self.update_layers):
            s, v = message_layer(s, v, edge, r_ij, distance)
            s, v = update_layer(s, v)
        
        s = self.last_mlp(s)
        return s

def build_edge_index(atom_positions, cutoff_distanceance, graph_indexes):
    edge_index =radius_graph(atom_positions, r=cutoff_distanceance, batch=graph_indexes, flow='target_to_source')
    return edge_index