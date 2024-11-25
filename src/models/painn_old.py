import torch
import torch.nn as nn
import numpy as np
import time
from scipy.spatial import distance_matrix


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
        device = 'cpu'
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
        self.device = device

        self.embedding_matrix = nn.Embedding(self.num_unique_atoms, self.num_features)

        self.message1 = Message(self.num_features, self.cutoff_dist, self.device)
        self.update1 = Update(self.num_features, self.device)

        self.message2 = Message(self.num_features, self.cutoff_dist, self.device)
        self.update2 = Update(self.num_features, self.device)
        
        self.message3 = Message(self.num_features, self.cutoff_dist, self.device)
        self.update3 = Update(self.num_features, self.device)

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
        
        v_0 = torch.zeros((atoms.shape[0], self.num_features, 3), device=self.device)  
        s_0 = self.embedding_matrix(atoms).to(self.device)
        
        print('Initial:')
        print("s: ", torch.mean(torch.abs(s_0)))
        print("v: ", torch.mean(torch.abs(v_0)))

        # Calculate vector between all nodes of same graph (molecule)
        start_time = time.time()
        r = self.calculate_rij(atom_positions, graph_indexes, self.cutoff_dist)
        #print("--- %s seconds ---" % (time.time() - start_time))

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
        print("--- %s seconds ---" % (time.time() - start_time))

        return E
        

    # REWRITE CALCULATE Rij to matrix format where i and j are row and column number and the element is the vector (three dimensions deep)
    def calculate_distance_matrix(self, atom_positions):
        atom_norm = (atom_positions ** 2).sum(dim=1, keepdim=True)  # Shape (N, 1)
        # Use broadcasting to compute the squared distance matrix
        dist_squared = atom_norm + atom_norm.T - 2 * torch.mm(atom_positions, atom_positions.T)
        # Clamp to avoid numerical precision issues and take square root
        dist = torch.sqrt(torch.clamp(dist_squared, min=10e-8))
        return dist

    def calculate_rij(self, atom_positions, graph_indexes, threshold):
        #dist_mat = distance_matrix(atom_positions.cpu(), atom_positions.cpu())
        dist_mat = self.calculate_distance_matrix(atom_positions)
        dist_mask = dist_mat < self.cutoff_dist
        #np.fill_diagonal(dist_mask, False)  # not bounded to itself
        dist_mask = dist_mask.fill_diagonal_(0)

        # set all atoms that are not in same molecule to False
        same_molecule_mask = (graph_indexes.unsqueeze(0) == graph_indexes.unsqueeze(1))  # (num_atoms, num_atoms)

        # final mask
        #final_mask  = torch.from_numpy(dist_mask).to(self.device) & same_molecule_mask
        final_mask  = dist_mask & same_molecule_mask

        pairs = torch.argwhere(final_mask) # works both ways!
        n_diff = atom_positions[pairs[:, 1]] - atom_positions[pairs[:, 0]]

        return torch.concat([pairs, n_diff], axis=1)

    
    def calculate_rij2(self, atom_positions, graph_indexes, threshold):
        #mol1: 0->1, 0->2, 0->3 ... 1->0, 1->2, 1->3 ...
        # what is missing is the index of the atoms and the adjacency matrix with 3 rows, index1, index2, vector rij.
        # Format:
        # Index i [0,0,0,0,..1,1,1,1,..2,2,2,2...]
        # Index j [0,1,2,...]
        # x for r
        # y for r
        # z for r

        index_i = torch.tensor([], device=self.device)
        index_j = torch.tensor([], device=self.device)
        r = 0

        j = torch.arange(len(atom_positions),  device=self.device)
        for i in range(len(atom_positions)):
            molecule_mask = graph_indexes == graph_indexes[i]
            js = j[molecule_mask]
            molecule_positions = atom_positions[molecule_mask]
            vectors = atom_positions[i] - molecule_positions

            distance_mask = torch.linalg.norm(vectors, axis=1) < threshold
            index_i = torch.cat([index_i, torch.tensor([int(i)] * sum(distance_mask),  device=self.device)], dim=0)
            index_j = torch.cat([index_j, js[distance_mask]], dim=0)
            if i ==0:
                r = vectors[distance_mask]
            else:
                r = torch.cat([r, vectors[distance_mask]])
        adjacency_matrix = torch.cat([index_i.unsqueeze(1), index_j.unsqueeze(1), r], dim=1)
        return adjacency_matrix

class Message(nn.Module):
    def __init__(self, num_features, cutoff_dist, device):
        super().__init__()
        self.num_features = num_features
        self.cutoff_dist = cutoff_dist
        self.device = device

        self.s_path = nn.Sequential( # rename
            nn.Linear(128, 128, True),
            nn.SiLU(),
            nn.Linear(128, 384))
        self.r_path = nn.Sequential(
            nn.Linear(20, 384, True))
        
            # Initialize weights
        for m in self.s_path.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        #torch.nn.init.kaiming_normal_(self.r_path[0].weight, nonlinearity='relu')
        #torch.nn.init.constant_(self.r_path[0].bias, 0)
        
    # def RBF(self, r, num_rbf_features=20):
    #     vector_r = r[:,2:]
    #     norm_r = torch.linalg.norm(vector_r, axis=1).unsqueeze(1) # normalize each r vector not all into 1 number
    #     frac = torch.pi / self.cutoff_dist
    #     n = torch.arange(1,num_rbf_features + 1, device=self.device).float().reshape(1, num_rbf_features)
    #     epsilon = 1e-8
    #     rbf_result = torch.sin(n * frac * norm_r) / (norm_r + epsilon)
    #     return rbf_result
    
    def RBF(self, r, num_rbf_features=20):
        norm_r = torch.linalg.norm(r, dim=1, keepdim=True)
        frac = torch.pi / self.cutoff_dist
        n = torch.arange(1, num_rbf_features + 1, device=self.device).float().reshape(1, num_rbf_features)
        epsilon = 1e-8
        norm_r = torch.clamp(norm_r, min=epsilon)  # Prevent division by zero
        rbf_result = torch.sin(n * frac * norm_r) / (norm_r + epsilon)  # Avoid dividing by zero
        rbf_result = torch.clamp(rbf_result, min=-1.0, max=1.0)
        return rbf_result
    
    def cosine_cutoff(self, r): # OBS DONT KNOW IF IT IS CUTOFF_DISTANCE OR ANOTHER CUTOFF PARAMETER
         # remember cosine cutoff: https://ml4chem.dev/_modules/ml4chem/atomistic/features/cutoff.html#Cosine has code
        cutoff_array = 0.5 * (torch.cos(torch.pi * r / self.cutoff_dist) + 1.0)
        cutoff_array *= (r < self.cutoff_dist).float()
        return cutoff_array
        
    def forward(self, v, s, r):
        js = r[:, 1].int() # r holds the following col: i, j, rx, ry, rz
        r_vectors = r[:,2:]
        #assert not torch.isnan(r_vectors).any(), "Message: Found NaN in r_vectors"
        #assert not (r_vectors == float('inf')).any(), "Message: Found Inf in r_vectors"
        #assert torch.all(r_vectors.abs() < 1e4), "Message: Found extreme values in r_vectors"

        phi = self.s_path(s)
        W = self.cosine_cutoff(self.r_path(self.RBF(r_vectors)))
        #rbf_result = self.RBF(r_vectors)
        #rbf_result = torch.clamp(rbf_result, min=-1.0, max=1.0)  # Normalize to [-1, 1]
        #assert not torch.isnan(rbf_result).any(), "Message: Found NaN in RBF output"
        #assert not (rbf_result == float('inf')).any(), "Message: Found Inf in RBF output"
        #assert torch.all(rbf_result.abs() < 1e6), "Message: Found extreme values in RBF output"
        #r_path_weights = self.r_path[0].weight  # First layer in Sequential
        #r_path_bias = self.r_path[0].bias
        #assert not torch.isnan(r_path_weights).any(), f"r_path: Found NaN in weights"
        #assert not torch.isnan(r_path_bias).any(), "r_path: Found NaN in biases"
        #assert torch.all(r_path_weights.abs() < 1e6), "r_path: Found extreme values in weights"

        #W = self.cosine_cutoff(self.r_path(rbf_result))
        #assert not torch.isnan(self.RBF(r_vectors)).any(), "Message: Found NaN in tensor result RBF"
        #assert not torch.isnan(self.r_path(self.RBF(r_vectors))).any(), "Message: Found NaN in tensor result RBF"
        #assert not torch.isnan(W).any(), "Message: Found NaN in tensor W"

        pre_split = phi[js, :] * W

        split_1 = pre_split[:,:self.num_features]
        split_2 = pre_split[:,self.num_features:self.num_features*2]
        split_3 = pre_split[:,self.num_features*2:]

        epsilon = 1e-8
        left_1 = v[js,:,:] * split_1.unsqueeze(2) 
        #right_1 = (split_3.unsqueeze(2) * (r_vectors / (torch.linalg.norm(r_vectors, axis=1).unsqueeze(1) + epsilon)).unsqueeze(1))
        r_norm = torch.linalg.norm(r_vectors, dim=1, keepdim=True)
        r_vectors_normalized = r_vectors / (r_norm + epsilon)
        right_1 = (split_3.unsqueeze(2) * (r_vectors_normalized / (torch.linalg.norm(r_vectors, axis=1).unsqueeze(1) + epsilon)).unsqueeze(1))
        left_2 = left_1 + right_1

        assert not torch.isnan(left_2).any(), "Message: Found NaN in tensor left_2"
        assert not torch.isnan(split_2).any(), "Message: Found NaN in tensor split_2"

        # Sum over j
        v1 = torch.zeros_like(v)
        delta_v = v1.index_add_(0, js, left_2)

        s1 = torch.zeros_like(s)
        delta_s = s1.index_add_(0, js, split_2)

        print('Message:')
        print("s: ", torch.mean(torch.abs(delta_s)))
        print("v: ", torch.mean(torch.abs(delta_v)))
        assert not torch.isnan(delta_s).any(), "Message: Found NaN in tensor delta_s"
        assert not torch.isinf(delta_s).any(), "Message: Found Inf in tensor delta_s"
        assert not torch.isnan(delta_v).any(), "Message: Found NaN in tensor delta_v"
        assert not torch.isinf(delta_v).any(), "Message: Found Inf in tensor delta_v"

        return delta_v, delta_s

class Update(nn.Module):
    def __init__(self, num_features, device):
        super().__init__()
        self.num_features = num_features
        self.device = device

        self.mlp_update = nn.Sequential(
            nn.Linear(num_features * 2, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features * 3)
        )

        self.U = nn.Linear(self.num_features, self.num_features)
        self.V = nn.Linear(self.num_features, self.num_features)
    
    def forward(self, v, s):
        # U pass
        v_permuted = torch.permute(v, (0,2,1))
        Uv = torch.permute(self.U(v_permuted), (0,2,1))
        Vv = torch.permute(self.V(v_permuted), (0,2,1))
        Vv_norm = torch.linalg.norm(Vv, dim=2)

        mlp_input = torch.hstack([Vv_norm, s])
        mlp_result = self.mlp_update(mlp_input)
        
        a_vv = mlp_result[:,:self.num_features]
        a_sv = mlp_result[:,self.num_features:self.num_features*2]
        a_ss = mlp_result[:,self.num_features*2:]

        delta_v = a_vv.unsqueeze(-1) * Uv
        
        dot_prod = torch.sum(Uv * Vv, dim=2) # dot product
        delta_s = dot_prod * a_sv + a_ss
        print('Update:')
        print("s: ", torch.mean(torch.abs(delta_s)))
        print("v: ", torch.mean(torch.abs(delta_v)))
        #delta_v = torch.clamp(delta_v, min=-1000, max=1000)
        #delta_s = torch.clamp(delta_s, min=-1000, max=1000)
    

        # assert not torch.isnan(delta_s).any(), "Update: Found NaN in tensor delta_s"
        # assert not torch.isinf(delta_s).any(), "Update: Found Inf in tensor delta_s"
        # assert not torch.isnan(delta_v).any(), "Update: Found NaN in tensor delta_v"
        # assert not torch.isinf(delta_v).any(), "Update: Found Inf in tensor delta_v"

        return delta_v, delta_s

