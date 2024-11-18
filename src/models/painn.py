import torch
import torch.nn as nn
import numpy as np


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
        self.num_rbf_features = num_rbf_features 
        self.num_unique_atoms = num_unique_atoms
        self.cutoff_dist = cutoff_dist

        self.embedding_matrix = nn.Embedding(self.num_unique_atoms, self.num_features)

        self.message1 = Message(self.num_features, self.cutoff_dist)
        self.update1 = Update(self.num_features)

        self.message2 = Message(self.num_features, self.cutoff_dist)
        self.update2 = Update(self.num_features)

        self.message3 = Message(self.num_features, self.cutoff_dist)
        self.update3 = Update(self.num_features)

        self.last_mlp = nn.Sequential(
            nn.Linear(self.num_features, self.num_features),
            nn.SiLU(),
            nn.Linear(self.num_features, self.num_outputs)
        )
        #raise NotImplementedError

    #def parameters(self):
    #    return []


    def forward(
        self,
        atoms: torch.LongTensor,
        atom_positions: torch.FloatTensor,
        graph_indexes: torch.LongTensor,
    ) -> torch.FloatTensor:
        """
        Forward pass of PaiNN. Includes the readout network highlighted in blue
        in Figure 2 in (Schütt et al., 2021) with normal U layers which is
        used for predicting properties as sums of atomic contributions. The
        post-processing and final sum is perfomed with
        src.models.AtomwisePostProcessing.

        Args:
            atoms: torch.LongTensor of size [num_nodes] with atom type of each
                node in the graph.
            atom_positions: torch.FloatTensor of size [num_nodes, 3] with
                euclidean coordinates of each node / atom.
            graph_indexes: torch.LongTensor of size [num_nodes] with the graph 
                index each node belongs to. (molecules the atoms belong to)

        Returns:
            A torch.FloatTensor of size [num_nodes, num_outputs] with atomic
            contributions to the overall molecular property prediction.
        """
        
        v_0 = torch.zeros((atoms.shape[0], self.num_features, 3))  
        s_0 = self.embedding_matrix(atoms)

        # Calculate vector between all nodes of same graph (molecule)
        r = self.calculate_rij(atom_positions, graph_indexes, self.cutoff_dist)
        

        v, s = self.message1(v_0, s_0, r)
        s_old = s + s_0 # skip connections
        v_old = v + v_0 # v_0 is a null vector - doesnt matter
        v, s = self.update1(v_old, s_old)
        s_old = s + s_old # skip connections
        v_old = v + v_old

        v, s = self.message2(v_old, s_old, r)
        s_old = s + s_old # skip connections
        v_old = v + v_old
        v, s = self.update2(v_old, s_old)
        s_old = s + s_old # skip connections
        v_old = v + v_old

        v, s = self.message3(v_old, s_old, r)
        s_old = s + s_old # skip connections
        v_old = v + v_old
        v, s = self.update3(v_old, s_old)
        s_old = s + s_old # skip connections
        v_old = v + v_old

        E = self.last_mlp(s_old)

        return E
        

    # REWRITE CALCULATE Rij to matrix format where i and j are row and column number and the element is the vector (three dimensions deep)
    def calculate_rij(self, atom_positions, graph_indexes, threshold):
        #mol1: 0->1, 0->2, 0->3 ... 1->0, 1->2, 1->3 ...
        # what is missing is the index of the atoms and the adjacency matrix with 3 rows, index1, index2, vector rij.
        # Format:
        # Index i [0,0,0,0,..1,1,1,1,..2,2,2,2...]
        # Index j [0,1,2,...]
        # x for r
        # y for r
        # z for r

        index_i = torch.tensor([])
        index_j = torch.tensor([])
        r = 0

        j = torch.arange(len(atom_positions))
        for i in range(len(atom_positions)):
            molecule_mask = graph_indexes == graph_indexes[i]
            js = j[molecule_mask]
            molecule_positions = atom_positions[molecule_mask]
            vectors = atom_positions[i] - molecule_positions

            distance_mask = torch.linalg.norm(vectors, axis=1) < threshold
            index_i = torch.cat([index_i, torch.tensor([int(i)] * sum(distance_mask))], dim=0)
            index_j = torch.cat([index_j, js[distance_mask]], dim=0)
            if i ==0:
                r = vectors[distance_mask]
            else:
                r = torch.cat([r, vectors[distance_mask]])
        adjacency_matrix = torch.cat([index_i.unsqueeze(1), index_j.unsqueeze(1), r], dim=1)
        return adjacency_matrix

class Message(nn.Module):
    def __init__(self, num_features, cutoff_dist):
        super().__init__()
        self.num_features = num_features
        self.cutoff_dist = cutoff_dist

        self.s_path = nn.Sequential( # rename
            nn.Linear(128, 128, True),
            nn.SiLU(),
            nn.Linear(128, 384))
        self.r_path = nn.Sequential(
            nn.Linear(20, 384, True))
        
    def RBF(self, r, num_rbf_features=20):
        vector_r = r[:,2:]
        norm_r = torch.linalg.norm(vector_r, axis=1).unsqueeze(1) # normalize each r vector not all into 1 number
        frac = torch.pi / self.cutoff_dist
        n = torch.arange(1,num_rbf_features+1).float().reshape(1, num_rbf_features)
        epsilon = 1e-8
        rbf_result = torch.sin(n * frac * norm_r) / (norm_r + epsilon)
        return rbf_result
    
    def cosine_cutoff(self, r): # OBS DONT KNOW IF IT IS CUTOFF_DISTANCE OR ANOTHER CUTOFF PARAMETER
         # remember cosine cutoff: https://ml4chem.dev/_modules/ml4chem/atomistic/features/cutoff.html#Cosine has code
        cutoff_array = 0.5 * (torch.cos(torch.pi * r / self.cutoff_dist) + 1.0)
        cutoff_array *= (r < self.cutoff_dist).float()
        return cutoff_array
        
    def forward(self, v, s, r):
        js = r[:, 1].int() # r holds the following col: i, j, rx, ry, rz
        r_vectors = r[:,2:]
        phi = self.s_path(s)
        W = self.cosine_cutoff(self.r_path(self.RBF(r_vectors)))

        pre_split = phi[js, :] * W

        split_1 = pre_split[:,:self.num_features]
        split_2 = pre_split[:,self.num_features:self.num_features*2]
        split_3 = pre_split[:,self.num_features*2:]

        epsilon = 1e-8
        left_1 = v[js,:,:] * split_1.unsqueeze(2) 
        right_1 = (split_3.unsqueeze(2) * (r_vectors / (torch.linalg.norm(r_vectors, axis=1).unsqueeze(1) + epsilon)).unsqueeze(1))
        left_2 = left_1 + right_1

        # Sum over j
        v1 = torch.zeros_like(v)
        delta_v = v1.index_add_(0, js, left_2)

        s1 = torch.zeros_like(s)
        delta_s = s1.index_add_(0, js, split_2)

        return delta_v, delta_s

class Update(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

        self.mlp_update = nn.Sequential(
            nn.Linear(num_features * 2, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features * 3)
        )

        self.U_x = nn.Linear(self.num_features, self.num_features)
        self.U_y = nn.Linear(self.num_features, self.num_features)
        self.U_z = nn.Linear(self.num_features, self.num_features)

        self.V_x = nn.Linear(self.num_features, self.num_features)
        self.V_y = nn.Linear(self.num_features, self.num_features)
        self.V_z = nn.Linear(self.num_features, self.num_features)
    
    def forward(self, v, s):
        # U pass
        v_x, v_y, v_z = v[:,:,0], v[:,:,1], v[:,:,2]
        Uv = torch.stack([self.U_x(v_x), self.U_y(v_y), self.U_z(v_z)], dim=2)
        Vv = torch.stack([self.V_x(v_x), self.V_y(v_y), self.V_z(v_z)], dim=2)
        Vv_norm = torch.linalg.norm(Vv, dim=2)

        mlp_input = torch.hstack([Vv_norm, s])
        mlp_result = self.mlp_update(mlp_input)
        
        a_vv = mlp_result[:,:self.num_features]
        a_sv = mlp_result[:,self.num_features:self.num_features*2]
        a_ss = mlp_result[:,self.num_features*2:]

        delta_v = a_vv.unsqueeze(-1) * Uv
        
        dot_prod = torch.sum(Uv * Vv, dim=2) # dot product
        delta_s = dot_prod * a_sv + a_ss

        return delta_v, delta_s

