"""
Graph Neural Network model for flow prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class FlowGNN(nn.Module):
    """
    Graph Neural Network for flow field prediction.
    Uses GCN layers with residual connections.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, 
                 output_dim: int = None, num_layers: int = 3,
                 dropout: float = 0.1, layer_type: str = 'GCN'):
        """
        Initialize FlowGNN model.
        
        Args:
            input_dim: Input feature dimension (e.g., 4 for velocity + pressure)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (default: same as input_dim for auto-regressive)
            num_layers: Number of GNN layers
            dropout: Dropout rate
            layer_type: Type of GNN layer ('GCN', 'GAT', 'GIN')
        """
        super(FlowGNN, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim else input_dim
        
        # Choose layer type
        if layer_type == 'GCN':
            LayerClass = GCNConv
        elif layer_type == 'GAT':
            LayerClass = lambda in_dim, out_dim: GATConv(in_dim, out_dim, heads=4, concat=False)
        elif layer_type == 'GIN':
            LayerClass = lambda in_dim, out_dim: GINConv(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim)
                )
            )
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.convs.append(LayerClass(hidden_dim, hidden_dim))
            else:
                self.convs.append(LayerClass(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output projection - initialize to predict small increments
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_dim)
        )
        
        # Initialize output layer to predict small increments initially
        # This helps the model start by predicting small changes
        nn.init.xavier_uniform_(self.output_proj[-1].weight, gain=0.1)
        nn.init.zeros_(self.output_proj[-1].bias)
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch vector for graph batching (optional)
        
        Returns:
            Output features [num_nodes, output_dim]
        """
        # Store input for residual connection
        input_x = x
        
        # Input projection
        x = self.input_proj(x)
        
        # GNN layers with residual connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            # Store residual
            residual = x
            
            # Apply convolution
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection (if dimensions match)
            if residual.shape == x.shape:
                x = x + residual
        
        # Output projection - predict increment
        delta = self.output_proj(x)
        
        # Add residual connection: output = input + delta
        # This allows the model to predict changes rather than absolute values
        if input_x.shape == delta.shape:
            output = input_x + delta
        else:
            # If dimensions don't match, just return delta (shouldn't happen)
            output = delta
        
        return output


class TemporalFlowGNN(nn.Module):
    """
    Temporal GNN for sequence-to-sequence flow prediction.
    Uses LSTM/GRU to model temporal dynamics.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 output_dim: int = None, num_layers: int = 3,
                 temporal_layers: int = 2, dropout: float = 0.1):
        """
        Initialize TemporalFlowGNN model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of GNN layers per time step
            temporal_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(TemporalFlowGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim else input_dim
        
        # Spatial GNN
        self.spatial_gnn = FlowGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Temporal LSTM
        self.temporal_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=temporal_layers,
            dropout=dropout if temporal_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection - initialize to predict small increments
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_dim)
        )
        
        # Initialize output layer to predict small increments initially
        nn.init.xavier_uniform_(self.output_proj[-1].weight, gain=0.1)
        nn.init.zeros_(self.output_proj[-1].bias)
    
    def forward(self, x_seq, edge_index, batch=None):
        """
        Forward pass for temporal sequence.
        
        Args:
            x_seq: Sequence of node features [seq_len, num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch vector (optional)
        
        Returns:
            Output sequence [seq_len, num_nodes, output_dim]
        """
        seq_len, num_nodes, _ = x_seq.shape
        
        # Process each time step through spatial GNN
        spatial_features = []
        for t in range(seq_len):
            spatial_out = self.spatial_gnn(x_seq[t], edge_index, batch)
            spatial_features.append(spatial_out)
        
        # Stack spatial features: [seq_len, num_nodes, hidden_dim]
        spatial_features = torch.stack(spatial_features, dim=0)
        
        # Reshape for LSTM: [num_nodes, seq_len, hidden_dim]
        spatial_features = spatial_features.transpose(0, 1)
        
        # Apply temporal LSTM
        lstm_out, _ = self.temporal_lstm(spatial_features)
        
        # Reshape back: [seq_len, num_nodes, hidden_dim]
        lstm_out = lstm_out.transpose(0, 1)
        
        # Output projection - predict increment
        delta = self.output_proj(lstm_out)
        
        # Add residual connection: output = last_input + delta
        # Predict next time step as increment from last input
        # We only care about the last time step prediction
        last_input = x_seq[-1]  # [num_nodes, input_dim]
        last_delta = delta[-1]  # [num_nodes, output_dim]
        
        if last_input.shape == last_delta.shape:
            # Predict next state as: current_state + delta
            next_state = last_input + last_delta
            # Return full sequence, but only last step has residual
            output = delta.clone()
            output[-1] = next_state
        else:
            # Fallback: just return delta (shouldn't happen)
            output = delta
        
        return output

