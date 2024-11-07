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
                index each node belongs to. (molecules the atoms belong to)

        Returns:
            A torch.FloatTensor of size [num_nodes, num_outputs] with atomic
            contributions to the overall molecular property prediction.
        """
        s_i_0 = self.embedding_matrix(atoms)
        v_i_0 = torch.zeros((atoms.shape[0], self.num_features, 3))        

        # Calculate vector between all nodes of same graph (molecule)
        r_ij= self.calculate_rij(atom_positions, graph_indexes, self.cutoff_dist)

        round1 = self.message(v_i_0, s_i_0, r_ij)


        # Cap on distance
        raise NotImplementedError
    
    def calculate_rij(self, atom_positions, graph_indexes, threshold):
        unique_molecules = graph_indexes.unique()
        vectors_per_molecule = [] #mol1: 0->1, 0->2, 0->3 ... 1->0, 1->2, 1->3 ...
        # what is missing is the index of the atoms and the adjacency matrix with 3 rows, index1, index2, vector rij.
        # Format:
        # Index i [0,0,0,0,..1,1,1,1,..2,2,2,2...]
        # Index j [0,1,2,...]
        # x for r_ij
        # y for r_ij
        # z for r_ij

        index_i = torch.tensor([])
        index_j = torch.tensor([])
        r_ij = 0

        j = torch.arange(len(atom_positions))
        for i in range(len(atom_positions)):
            molecule_mask = graph_indexes == graph_indexes[0]
            js = j[molecule_mask]
            molecule_positions = atom_positions[molecule_mask]
            vectors = atom_positions[i] - molecule_positions

            distance_mask = torch.linalg.norm(vectors, axis=1) < threshold
            index_i = torch.cat([index_i, torch.tensor([int(i)] * sum(distance_mask))], dim=0)
            index_j = torch.cat([index_j, js[distance_mask]], dim=0)
            if i ==0:
                r_ij = vectors[distance_mask]
            else:
                r_ij = torch.cat([r_ij, vectors[distance_mask]])
        adjacency_matrix = torch.cat([index_i.unsqueeze(1), index_j.unsqueeze(1), r_ij], dim=1)
        return adjacency_matrix
    
    def RBF(self, r_ij):
        norm_r_ij = torch.linalg.norm(r_ij)
        frac = torch.pi / self.cutoff_dist
        rbf_result = torch.sin(torch.arange(1,21) * frac * norm_r_ij) / norm_r_ij
        return rbf_result
        
    def message(self, v_j, s_j, r_ij):
        s_path = nn.Sequential(
            nn.Linear(128, 128, True),
            nn.SiLU(),
            nn.Linear(128, 384))
        rbf_output = self.RBF(r_ij)
        # remember cosine cutoff: https://ml4chem.dev/_modules/ml4chem/atomistic/features/cutoff.html#Cosine has code
        
        # Should this be made as an init and forward as well? Not necessary

        pass