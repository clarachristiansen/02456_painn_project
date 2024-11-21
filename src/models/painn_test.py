import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def gaussian_rbf(distances, num_rbf, cutoff_dist):
    """
    Expand distances into radial basis functions.
    
    Args:
        distances: Tensor of distances [num_edges]
        num_rbf: Number of basis functions
        cutoff_dist: Cutoff distance
    """
    means = torch.linspace(0, cutoff_dist, num_rbf, device=distances.device)
    sigma = means[1] - means[0]
    expanded = torch.exp(-((distances.unsqueeze(-1) - means) ** 2) / (2 * sigma ** 2))
    return expanded * (distances.unsqueeze(-1) < cutoff_dist)

class MessageBlock(nn.Module):
    def __init__(self, num_features, num_rbf_features):
        super().__init__()
        self.num_features = num_features
        
        # Message passing layers
        self.msg_scalar = nn.Sequential(
            nn.Linear(num_features + num_rbf_features, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features)
        )
        
        self.msg_vector = nn.Sequential(
            nn.Linear(num_features + num_rbf_features, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features)
        )
        
        # Update layers
        self.upd_scalar = nn.Sequential(
            nn.Linear(2 * num_features, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features)
        )
        
        self.upd_vector = nn.Sequential(
            nn.Linear(2 * num_features, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features)
        )
        
    def forward(self, s, v, edge_index, rbf):
        row, col = edge_index
        
        # Compute direction vectors between atoms
        v_diff = v[row] - v[col]  # [num_edges, 3, num_features]
        v_abs = torch.sqrt(torch.sum(v_diff ** 2, dim=1) + 1e-8)  # [num_edges, num_features]
        
        # Compute messages using RBF features
        s_msg = torch.cat([s[row], rbf], dim=-1)  # [num_edges, num_features + num_rbf]
        v_msg = torch.cat([s[row].unsqueeze(1).expand(-1, 3, -1), 
                          rbf.unsqueeze(1).expand(-1, 3, -1)], dim=-1)  # [num_edges, 3, num_features + num_rbf]
        
        # Transform messages
        ds = self.msg_scalar(s_msg)  # [num_edges, num_features]
        dv = self.msg_vector(v_msg.view(-1, self.num_features + rbf.size(-1)))  # [num_edges * 3, num_features]
        dv = dv.view(-1, 3, self.num_features)  # [num_edges, 3, num_features]
        
        # Normalize direction vectors
        v_diff_norm = v_diff / (torch.norm(v_diff, dim=1, keepdim=True) + 1e-8)  # [num_edges, 3, num_features]
        
        # Scale vector messages by normalized direction vectors
        dv = dv * v_diff_norm  # [num_edges, 3, num_features]
        
        # Aggregate messages
        ds_agg = torch.zeros_like(s).index_add_(0, col, ds)  # [num_nodes, num_features]
        dv_agg = torch.zeros_like(v).index_add_(0, col, dv)  # [num_nodes, 3, num_features]
        
        # Update scalar and vector features
        s_upd = torch.cat([s, ds_agg], dim=-1)  # [num_nodes, 2*num_features]
        v_upd = torch.cat([v, dv_agg], dim=-1)  # [num_nodes, 3, 2*num_features]
        
        # Transform updates
        s_out = s + self.upd_scalar(s_upd)  # [num_nodes, num_features]
        v_out = v + self.upd_vector(v_upd.reshape(-1, 2 * self.num_features)).view(-1, 3, self.num_features)  # [num_nodes, 3, num_features]
        
        return s_out, v_out

class PaiNN(nn.Module):
    def __init__(self, 
                 num_message_passing_layers,
                 num_features,
                 num_outputs,
                 num_rbf_features,
                 num_unique_atoms,
                 cutoff_dist,
                 device):
        super().__init__()
        self.num_features = num_features
        self.cutoff_dist = cutoff_dist
        self.num_rbf_features = num_rbf_features
        self.device = device
        
        # Initial embeddings
        self.embedding = nn.Embedding(num_unique_atoms, num_features)
        self.init_vector = nn.Linear(num_features, 3 * num_features)
        
        # Message passing blocks
        self.blocks = nn.ModuleList([
            MessageBlock(num_features, num_rbf_features) 
            for _ in range(num_message_passing_layers)
        ])
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_outputs)
        )
        
        self.to(device)
        
    def forward(self, atoms, atom_positions, graph_indexes):
        """
        Forward pass of PaiNN model.
        
        Args:
            atoms: torch.LongTensor of size [num_nodes] with atom type of each
                node in the graph.
            atom_positions: torch.FloatTensor of size [num_nodes, 3] with
                euclidean coordinates of each node / atom.
            graph_indexes: torch.LongTensor of size [num_nodes] with the graph 
                index each node belongs to. (molecules the atoms belong to)
                
        Returns:
            Predicted properties [num_graphs, num_outputs]
        """
        # Build edge index and compute distances
        edge_index = build_edge_index(atom_positions, self.cutoff_dist, graph_indexes)
        row, col = edge_index
        
        # Compute distances and RBF expansion
        distances = torch.norm(atom_positions[row] - atom_positions[col], dim=-1)
        rbf = gaussian_rbf(distances, self.num_rbf_features, self.cutoff_dist)
        
        # Initial embeddings
        s = self.embedding(atoms)  # [num_nodes, num_features]
        v = self.init_vector(s).view(-1, 3, self.num_features)  # [num_nodes, 3, num_features]
        
        # Message passing
        for block in self.blocks:
            s, v = block(s, v, edge_index, rbf)
        
        # Compute atomic contributions
        atomic_output = self.output_layers(s)  # [num_nodes, num_outputs]
        
        # Aggregate per molecule using graph_indexes
        #num_graphs = graph_indexes.max().item() + 1
        #output = torch.zeros(num_graphs, atomic_output.size(-1),
        #                   device=self.device)
        #output.index_add_(0, graph_indexes, atomic_output)
            
        return atomic_output

def build_edge_index(pos, cutoff, graph_indexes):
    """
    Build edge index based on distance cutoff using vectorized operations.
    
    Args:
        pos (torch.Tensor): Atomic positions [num_nodes, 3]
        cutoff (float): Distance cutoff for creating edges
        graph_indexes (torch.Tensor): Graph assignments for atoms [num_nodes]
    
    Returns:
        torch.Tensor: Edge index [2, num_edges]
    """
    num_nodes = pos.shape[0]
    
    # Create all possible pairs of indices
    node_indices = torch.arange(num_nodes, device=pos.device)
    rows, cols = torch.meshgrid(node_indices, node_indices, indexing='ij')
    rows, cols = rows.flatten(), cols.flatten()
    
    # Remove self-loops
    mask = rows != cols
    rows, cols = rows[mask], cols[mask]
    
    # Calculate pairwise distances
    distances = torch.norm(pos[rows] - pos[cols], dim=-1)
    
    # Apply distance cutoff
    mask = distances < cutoff
    rows, cols = rows[mask], cols[mask]
    
    # Only create edges within same molecule using graph_indexes
    same_molecule = graph_indexes[rows] == graph_indexes[cols]
    rows, cols = rows[same_molecule], cols[same_molecule]
    
    edge_index = torch.stack([rows, cols], dim=0)
    return edge_index

if __name__ == "__main__":
    # Example usage with all parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    class Args:
        num_message_passing_layers = 3
        num_features = 128
        num_outputs = 1
        num_rbf_features = 20
        num_unique_atoms = 100
        cutoff_dist = 5.0
    
    args = Args()
    
    # Create model
    model = PaiNN(
        num_message_passing_layers=args.num_message_passing_layers,
        num_features=args.num_features,
        num_outputs=args.num_outputs,
        num_rbf_features=args.num_rbf_features,
        num_unique_atoms=args.num_unique_atoms,
        cutoff_dist=args.cutoff_dist,
        device=device
    )
    
    # Example with two water molecules
    atoms = torch.tensor([8, 1, 1, 8, 1, 1], device=device)  # O, H, H, O, H, H
    atom_positions = torch.tensor([
        [0.0, 0.0, 0.0],    # First water molecule
        [0.0, 0.757, 0.586],
        [0.0, -0.757, 0.586],
        [2.0, 0.0, 0.0],    # Second water molecule
        [2.0, 0.757, 0.586],
        [2.0, -0.757, 0.586]
    ], device=device)
    graph_indexes = torch.tensor([0, 0, 0, 1, 1, 1], device=device)
    
    # Forward pass
    energies = model(atoms, atom_positions, graph_indexes)
    print(f"Predicted energies: {energies}")