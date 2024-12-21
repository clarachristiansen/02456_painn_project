import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph
from schnetpack.nn.radial import BesselRBF
from schnetpack.nn.cutoff import CosineCutoff
import numpy as np

class MessageBlock(nn.Module):
    def __init__(self, num_features, num_rbf_features, cutoff_dist):
        super().__init__()
        self.num_features = num_features
        self.num_rbf_features = num_rbf_features

        self.cutoff_dist = cutoff_dist

        # Message
        self.cutoff_function = CosineCutoff(self.cutoff_dist)

        self.phi_path = nn.Sequential(
            nn.Linear(self.num_features, self.num_features),
            nn.SiLU(),
            nn.Linear(self.num_features, self.num_features * 3))
        self.W_path = nn.Sequential(
            nn.Linear(self.num_rbf_features, self.num_features * 3))
        
        # Update
        self.U = nn.Linear(self.num_features, self.num_features)
        self.V = nn.Linear(self.num_features, self.num_features)

        self.mlp_update = nn.Sequential(
            nn.Linear(self.num_features * 2, self.num_features),
            nn.SiLU(),
            nn.Linear(self.num_features, self.num_features * 3)
        )

    def forward(self, s, v, i_index, j_index, rbf, r_ij_direction):
        # Message 
        phi = self.phi_path(s[j_index]) # [620, 384]
        W = self.cutoff_function(self.W_path(rbf)) #[620, 384]
        split = phi * W # i_index [620, 384]
        Ws, Wvs, Wvv = torch.split(split, self.num_features, dim=-1) # 3 * [620, 128]
        
        delta_v_all = Wvs.unsqueeze(-1) * r_ij_direction.unsqueeze(1) + Wvv.unsqueeze(-1) * v[j_index] # right and left path [620, 128, 3]
        
        delta_v = torch.zeros_like(v)
        delta_v = delta_v.index_add_(0, i_index, delta_v_all)
        v = v + delta_v

        delta_s = torch.zeros_like(s)
        delta_s = delta_s.index_add_(0, i_index, Ws)
        s = s + delta_s

        # Update 
        v_permuted = torch.permute(v, (0,2,1))
        Uv = torch.permute(self.U(v_permuted), (0,2,1))
        Vv = torch.permute(self.V(v_permuted), (0,2,1))
        Vv_norm = torch.linalg.norm(Vv, dim=2)
        mlp_input = torch.hstack([Vv_norm, s])
        mlp_result = self.mlp_update(mlp_input)

        a_vv, a_sv, a_ss = torch.split(mlp_result, self.num_features, dim=-1)
        
        dv = a_vv.unsqueeze(-1) * Uv
        
        dot_prod = torch.sum(Uv * Vv, dim=2) # dot product
        ds = dot_prod * a_sv + a_ss
        
        s = s + ds
        v = v + dv

        return s, v



class PaiNN(nn.Module):
    """
    Polarizable Atom Interaction Neural Network with PyTorch.
    """
    def __init__(
        self,
        num_message_passing_layers: int = 3,
        num_features: int = 128,
        num_outputs: int = 1,
        num_rbf_features: int = 20,
        num_unique_atoms: int = 100,
        cutoff_dist: float = 5.0,
        device: str='cpu',
    ) -> None:
        """
        Args:
            num_message_passing_layers: Number of message passing layers in
                the PaiNN model.
            num_features: Size of the node embeddings (scalar features) and
                vector features.
            num_outputs: Number of model outputs. In most cases 1.
            num_rbf_features: Number of radial basis functions to represent
                distances.
            num_unique_atoms: Number of unique atoms in the data that we want
                to learn embeddings for.
            cutoff_dist: Euclidean distance threshold for determining whether 
                two nodes (atoms) are neighbours.
        """
        super().__init__()
        self.num_message_passing_layers = num_message_passing_layers
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.cutoff_dist = cutoff_dist
        self.num_rbf_features = num_rbf_features
        self.num_unique_atoms = num_unique_atoms
        self.device = device
        
        self.to(device)

        # Initial embeddings function
        self.embedding = nn.Embedding(self.num_unique_atoms, self.num_features)    
        # RBF
        self.rbf = BesselRBF(self.num_rbf_features, self.cutoff_dist)

        # Message blocks (both message and update)
        self.blocks = nn.ModuleList([
            MessageBlock(num_features, num_rbf_features, self.cutoff_dist) 
            for _ in range(self.num_message_passing_layers)
        ])

        # Last MLP
        self.last_mlp = nn.Sequential(
            nn.Linear(self.num_features, self.num_features),
            nn.SiLU(),
            nn.Linear(self.num_features, self.num_outputs)
        )

    def forward(
        self,
        atoms: torch.LongTensor,
        atom_positions: torch.FloatTensor,
        graph_indexes: torch.LongTensor,
    ) -> torch.FloatTensor:
        """
        Forward pass of PaiNN. Includes the readout network highlighted in blue
        in Figure 2 in (Schütt et al., 2021) with normal linear layers which is
        used for predicting properties as sums of atomic contributions. The
        post-processing and final sum is perfomed with
        src.models.AtomwisePostProcessing.

        Args:
            atoms: torch.LongTensor of size [num_nodes] with atom type of each
                node in the graph.
            atom_positions: torch.FloatTensor of size [num_nodes, 3] with
                euclidean coordinates of each node / atom.
            graph_indexes: torch.LongTensor of size [num_nodes] with the graph 
                index each node belongs to.

        Returns:
            A torch.FloatTensor of size [num_nodes, num_outputs] with atomic
            contributions to the overall molecular property prediction.
        """
        s = self.embedding(atoms)
        v = torch.zeros((atoms.shape[0], self.num_features, 3), device=self.device) 
        i_index, j_index = build_edge_index(atom_positions, self.cutoff_dist, graph_indexes)
        r_ij = atom_positions[j_index] - atom_positions[i_index]
        distance = torch.linalg.norm(r_ij, axis=1, keepdim=True)
        #distance = torch.clamp(distance, min=1e-8)
        rbf = self.rbf(distance.squeeze())
        r_ij_direction = r_ij / (distance + 1e-8)
        # Message passing
        for block in self.blocks:
            s, v = block(s, v, i_index, j_index, rbf, r_ij_direction)
        
        E = self.last_mlp(s)
        #print(E)
        return E
        

def build_edge_index(atom_positions, cutoff_distance, graph_indexes):
    edge_index =radius_graph(atom_positions, r=cutoff_distance, batch=graph_indexes, flow='target_to_source')
    #print(edge_index)
    return edge_index