"""
Enhanced dataset with flattened sequence input format.
Based on cylinder project methodology.
"""
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from typing import List
import numpy as np


class EnhancedFlowDataset(Dataset):
    """
    Dataset that flattens sequence into single input vector per node.
    
    Input format: x_i = [u_i[t1:t6], v_i[t1:t6], p_i[t1:t6], ...]
    Output format: y_i = [u_i[t7], v_i[t7], p_i[t7], ...]
    """
    
    def __init__(self, graphs: List, sequence_length: int = 6, predict_ahead: int = 1):
        """
        Initialize enhanced dataset.
        
        Args:
            graphs: List of graph Data objects for each time step
            sequence_length: Number of input time steps (default: 6)
            predict_ahead: Number of steps ahead to predict (default: 1)
        """
        self.graphs = graphs
        self.sequence_length = sequence_length
        self.predict_ahead = predict_ahead
    
    def __len__(self):
        return len(self.graphs) - self.sequence_length - self.predict_ahead + 1
    
    def __getitem__(self, idx):
        # Get input sequence: graphs at t1 to t6
        input_graphs = self.graphs[idx:idx + self.sequence_length]
        
        # Get target: graph at t7 (idx + sequence_length + predict_ahead - 1)
        target_idx = idx + self.sequence_length + self.predict_ahead - 1
        target_graph = self.graphs[target_idx]
        
        # Flatten sequence: concatenate features from all time steps
        # Format: [u[t1:t6], v[t1:t6], p[t1:t6], ...]
        flattened_features = []
        for graph in input_graphs:
            # Extract fields: velocity (u, v, w) and pressure (p)
            features = graph.x  # [num_nodes, features]
            flattened_features.append(features)
        
        # Concatenate along feature dimension: [num_nodes, features * time_steps]
        input_features = torch.cat(flattened_features, dim=1)
        
        # Create new graph with flattened features
        # Use edge_index from first graph (same for all time steps)
        input_graph = type(input_graphs[0])(
            x=input_features,
            edge_index=input_graphs[0].edge_index,
            num_nodes=input_graphs[0].num_nodes
        )
        
        return input_graph, target_graph


def enhanced_collate_fn(batch):
    """Custom collate function for enhanced dataset."""
    input_graphs, target_graphs = zip(*batch)
    
    # Batch input graphs
    batched_inputs = Batch.from_data_list(input_graphs)
    
    # Batch targets
    batched_targets = Batch.from_data_list(target_graphs)
    
    return batched_inputs, batched_targets

