"""
GNN model following exact method specification.

Input: x_i^(t-1:t-6) = [c_i, p_i, u_i, v_i]_(t-1:t-6)  (6 time steps, flattened)
Output: y_i^t = [c_i, p_i, u_i, v_i]_t  (next time step)

Architecture:
- Single-head GAT layers
- Multi-head GAT layers (K heads)
- Residual connections: y^l = x^l + F(x^l, W^l), x^(l+1) = σ(y^l)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_method import GATLayerMethod, MultiHeadGATMethod


class FlowGNNMethod(nn.Module):
    """
    GNN model following exact method specification.
    
    Learned simulator: y_i^t = Σ_{j∈ne(i)} f_ω(x_i^(t-1:t-6), x_j^(t-1:t-6), e_ij)
    
    Architecture:
    - Input: Flattened sequence [c, p, u, v]_(t-1:t-6) = 4 * 6 = 24 features
    - Output: Next time step [c, p, u, v]_t = 4 features
    - Single-head + Multi-head GAT layers
    - Residual connections
    """
    
    def __init__(self, input_dim: int = 24, hidden_dim: int = 64, 
                 output_dim: int = 4, num_layers: int = 4,
                 dropout: float = 0.1):
        """
        Initialize FlowGNN following method specification.
        
        Args:
            input_dim: Input dimension (4 fields * 6 time steps = 24)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (4 fields: c, p, u, v)
            num_layers: Number of GNN layers
            dropout: Dropout rate
        """
        super(FlowGNNMethod, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GAT layers: alternate between single-head and multi-head
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                # First layer: single-head GAT
                self.layers.append(GATLayerMethod(hidden_dim, hidden_dim, dropout))
            elif i % 2 == 1:
                # Odd layers: multi-head GAT (4 heads)
                self.layers.append(MultiHeadGATMethod(hidden_dim, hidden_dim, num_heads=4, dropout=dropout))
            else:
                # Even layers: single-head GAT
                self.layers.append(GATLayerMethod(hidden_dim, hidden_dim, dropout))
            
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # LeakyReLU activation (σ)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = dropout
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass following method specification.
        
        Args:
            x: Node features [num_nodes, input_dim]
               Format: [c, p, u, v]_(t-1:t-6) flattened = 24 features
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch vector (optional)
        
        Returns:
            Output features [num_nodes, output_dim]
            Format: [c, p, u, v]_t = 4 features
        """
        # Store input for residual connection
        input_x = x
        
        # Input projection
        x = self.input_proj(x)
        
        # Store first layer input for residual
        x_0 = x
        
        # GAT layers with residual connections
        # Residual pattern: y^l = x^l + F(x^l, W^l), x^(l+1) = σ(y^l)
        for i, (layer, bn) in enumerate(zip(self.layers, self.batch_norms)):
            residual = x
            
            # Apply GAT layer: F(x^l, W^l)
            x = layer(x, edge_index)
            x = bn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection: y^l = x^l + F(x^l, W^l)
            if residual.shape == x.shape:
                x = residual + x
            
            # Apply activation: x^(l+1) = σ(y^l)
            x = self.leaky_relu(x)
        
        # Output projection
        output = self.output_proj(x)
        
        # Add residual connection from input (last time step)
        # Extract last time step from flattened input: [c, p, u, v] at t-6
        if input_x.shape[1] == 24:  # 4 fields * 6 time steps
            last_timestep = input_x[:, -4:]  # Last 4 features = [c, p, u, v] at t-6
            if last_timestep.shape == output.shape:
                # Predict increment: output = last_timestep + delta
                output = last_timestep + output
        
        return output


class TemporalFlowGNNMethod(nn.Module):
    """
    Temporal GNN following method specification.
    
    Uses spatial GNN + temporal modeling for sequence prediction.
    """
    
    def __init__(self, input_dim: int = 24, hidden_dim: int = 64,
                 output_dim: int = 4, num_layers: int = 4,
                 temporal_layers: int = 2, dropout: float = 0.1):
        """
        Initialize Temporal FlowGNN.
        
        Args:
            input_dim: Input dimension (4 fields * 6 time steps = 24)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (4 fields: c, p, u, v)
            num_layers: Number of spatial GNN layers
            temporal_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(TemporalFlowGNNMethod, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Spatial GNN
        self.spatial_gnn = FlowGNNMethod(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,  # Output hidden features for LSTM
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
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass for temporal model.
        
        Args:
            x: Node features [num_nodes, input_dim]
               Format: [c, p, u, v]_(t-1:t-6) flattened = 24 features
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch vector (optional)
        
        Returns:
            Output features [num_nodes, output_dim]
            Format: [c, p, u, v]_t = 4 features
        """
        # Process through spatial GNN
        spatial_out = self.spatial_gnn(x, edge_index, batch)
        
        # Reshape for LSTM: [num_nodes, 1, hidden_dim]
        spatial_out = spatial_out.unsqueeze(1)
        
        # Apply temporal LSTM
        lstm_out, _ = self.temporal_lstm(spatial_out)
        
        # Reshape: [num_nodes, hidden_dim]
        lstm_out = lstm_out.squeeze(1)
        
        # Output projection
        output = self.output_proj(lstm_out)
        
        # Add residual from last time step
        if x.shape[1] == 24:
            last_timestep = x[:, -4:]
            if last_timestep.shape == output.shape:
                output = last_timestep + output
        
        return output

