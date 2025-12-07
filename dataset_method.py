"""
Dataset for method-based GNN training.

Input: 6 previous time steps: x_i^(t-1:t-6) = [c_i, p_i, u_i, v_i]_(t-1:t-6)
Output: Next time step: y_i^t = [c_i, p_i, u_i, v_i]_t
"""
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class FlowDatasetMethod(Dataset):
    """
    Dataset following exact method specification.
    
    Input: 6 previous time steps flattened per node
    Output: Next time step
    """
    
    def __init__(self, graphs, sequence_length=6, predict_ahead=1):
        """
        Initialize dataset.
        
        Args:
            graphs: List of graph Data objects with features [c, p, u, v]
            sequence_length: Number of input time steps (must be 6 per method)
            predict_ahead: Number of steps ahead to predict
        """
        if sequence_length != 6:
            print(f"Warning: Method specifies 6 time steps, but got {sequence_length}. Using 6.")
            sequence_length = 6
        
        self.graphs = graphs
        self.sequence_length = 6  # Fixed to 6 per method
        self.predict_ahead = predict_ahead
    
    def __len__(self):
        return len(self.graphs) - self.sequence_length - self.predict_ahead + 1
    
    def __getitem__(self, idx):
        # Get input sequence: 6 previous time steps
        input_graphs = self.graphs[idx:idx + self.sequence_length]
        
        # Get target: next time step (t+1)
        target_idx = idx + self.sequence_length + self.predict_ahead - 1
        target_graph = self.graphs[target_idx]
        
        # Flatten sequence per node: [c, p, u, v]_(t-1:t-6)
        # Each graph has features [c, p, u, v] = 4 features
        # Flatten: [c[t-6], p[t-6], u[t-6], v[t-6], ..., c[t-1], p[t-1], u[t-1], v[t-1]]
        flattened_features = []
        for graph in input_graphs:
            # graph.x shape: [num_nodes, 4] = [c, p, u, v]
            flattened_features.append(graph.x)
        
        # Concatenate along feature dimension: [num_nodes, 4 * 6] = [num_nodes, 24]
        input_features = torch.cat(flattened_features, dim=1)
        
        # Create new graph with flattened features
        input_graph = type(input_graphs[0])(
            x=input_features,
            edge_index=input_graphs[0].edge_index,
            num_nodes=input_graphs[0].num_nodes
        )
        
        return input_graph, target_graph


def collate_fn_method(batch):
    """Custom collate function for method-based dataset."""
    input_graphs, target_graphs = zip(*batch)
    
    # Batch input graphs
    batched_inputs = Batch.from_data_list(input_graphs)
    
    # Batch targets
    batched_targets = Batch.from_data_list(target_graphs)
    
    return batched_inputs, batched_targets

