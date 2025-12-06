"""
Enhanced GNN model with explicit neighbor aggregation and flattened sequence input.
Based on cylinder project methodology.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.utils import scatter
from attention_layers import SingleHeadAttention, MultiHeadAttention


class EnhancedFlowGNN(nn.Module):
    """
    Enhanced GNN model with custom attention mechanism.
    
    Architecture:
    - Single-headed attention layer
    - Multi-headed attention layer (4 heads, K and Q dim=1)
    - Residual connections: between multi-head and single-head, and every 2 layers
    - Deep network to capture k-order neighbors
    
    Input: Flattened sequence x_i = [u_i[t1:t6], v_i[t1:t6], p_i[t1:t6], ...]
    Output: Next time step y_i = [u_i[t7], v_i[t7], p_i[t7], ...]
    
    Function: y_i = f(x_i[t1:t6], sum(x_j[t1:t6] for j in neighbors[i]), e_ij, omega)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, 
                 output_dim: int = None, num_layers: int = 3,
                 dropout: float = 0.1, layer_type: str = 'attention',
                 use_explicit_aggregation: bool = True):
        """
        Initialize Enhanced FlowGNN model.
        
        Args:
            input_dim: Input feature dimension (flattened sequence: fields * time_steps)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (default: fields only, not time sequence)
            num_layers: Number of GNN layers
            dropout: Dropout rate
            layer_type: Type of layer ('attention', 'GCN', 'GAT', 'GIN')
            use_explicit_aggregation: If True, explicitly aggregate neighbor features
        """
        super(EnhancedFlowGNN, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_explicit_aggregation = use_explicit_aggregation
        
        # Output dimension: typically fields only (u, v, p) = 3 + 1 = 4
        # Input is flattened sequence, output is single time step
        self.output_dim = output_dim if output_dim else (input_dim // 6)  # Assume 6 time steps
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Explicit neighbor aggregation layer (optional)
        if use_explicit_aggregation:
            self.aggregation_proj = nn.Linear(input_dim, hidden_dim)
        
        # Attention layers: single-head and multi-head
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        
        if layer_type == 'attention':
            # Custom attention architecture
            self.attention_layers = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            
            for i in range(num_layers):
                # First layer: single-head attention
                if i == 0:
                    self.attention_layers.append(
                        SingleHeadAttention(hidden_dim, hidden_dim, dropout)
                    )
                    self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
                else:
                    # Alternate between multi-head and single-head
                    if i % 2 == 1:
                        # Multi-head attention (4 heads)
                        self.attention_layers.append(
                            MultiHeadAttention(hidden_dim, hidden_dim, num_heads=4, dropout=dropout)
                        )
                    else:
                        # Single-head attention
                        self.attention_layers.append(
                            SingleHeadAttention(hidden_dim, hidden_dim, dropout)
                        )
                    self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        else:
            # Fallback to standard layers
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
            
            self.attention_layers = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            for i in range(num_layers):
                self.attention_layers.append(LayerClass(hidden_dim, hidden_dim))
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_dim)
        )
        
        # Initialize output layer for small increments
        nn.init.xavier_uniform_(self.output_proj[-1].weight, gain=0.1)
        nn.init.zeros_(self.output_proj[-1].bias)
        
        self.dropout = dropout
        self.layer_type = layer_type
    
    def aggregate_neighbors(self, x, edge_index):
        """
        Explicitly aggregate neighbor features: sum(x_j for j in neighbors[i]).
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
        
        Returns:
            Aggregated neighbor features [num_nodes, input_dim]
        """
        # For each node, sum features of its neighbors
        row, col = edge_index
        neighbor_sum = scatter(x[col], row, dim=0, dim_size=x.size(0), reduce='sum')
        
        # Count neighbors for normalization (optional)
        neighbor_count = scatter(torch.ones_like(col, dtype=torch.float), row, 
                                dim=0, dim_size=x.size(0), reduce='sum')
        neighbor_count = neighbor_count.unsqueeze(1) + 1e-8  # Avoid division by zero
        
        # Normalized aggregation (mean)
        neighbor_mean = neighbor_sum / neighbor_count
        
        return neighbor_mean
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass with explicit neighbor aggregation.
        
        Args:
            x: Node features [num_nodes, input_dim] (flattened sequence)
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch vector (optional)
        
        Returns:
            Output features [num_nodes, output_dim] (single time step)
        """
        # Store input for residual connection
        input_x = x
        
        # Explicit neighbor aggregation: sum(x_j for j in neighbors[i])
        if self.use_explicit_aggregation:
            neighbor_features = self.aggregate_neighbors(x, edge_index)
            # Project aggregated neighbor features
            neighbor_proj = self.aggregation_proj(neighbor_features)
        else:
            neighbor_proj = None
        
        # Project input features
        x = self.input_proj(x)
        
        # Combine with neighbor aggregation if used
        if neighbor_proj is not None:
            x = x + neighbor_proj  # Combine node and neighbor information
        
        # Store input for residual connection from first layer
        x_0 = x
        
        # Attention layers with residual connections
        # Residual pattern: first layer, then every 2 layers
        # Also between multi-head and single-head
        for i, (attn_layer, bn) in enumerate(zip(self.attention_layers, self.batch_norms)):
            residual = x
            
            # Apply attention layer
            x = attn_layer(x, edge_index)
            x = bn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connections:
            # 1. First layer: y_0 = x_0 + F(x_0, W_0)
            # 2. Every 2 layers after first: y_l = x_l + F(x_l, W_l)
            # 3. Between multi-head and single-head (handled by alternating pattern)
            if i == 0:
                # First layer: residual from input
                if x.shape == x_0.shape:
                    x = x_0 + x
            elif i % 2 == 0:
                # Every 2 layers (0-indexed: layers 2, 4, 6, ...)
                if residual.shape == x.shape:
                    x = residual + x
            elif i % 2 == 1 and i > 1:
                # Between multi-head (i-1) and single-head (i)
                # Residual from previous layer
                if residual.shape == x.shape:
                    x = residual + x
            
            # Apply activation (LeakyReLU)
            x = self.leaky_relu(x)
        
        # Output projection - predict increment
        delta = self.output_proj(x)
        
        # Add residual connection: output = last_input_state + delta
        # Extract last time step from flattened input for residual
        # Assuming input is [u[t1:t6], v[t1:t6], p[t1:t6], ...]
        # We want to predict from the last time step values
        if input_x.shape[1] == self.output_dim * 6:  # 6 time steps
            # Extract last time step (fields at t6)
            last_timestep = input_x[:, -self.output_dim:]
            if last_timestep.shape == delta.shape:
                output = last_timestep + delta
            else:
                output = delta
        else:
            # If dimensions don't match, just return delta
            output = delta
        
        return output


class EnhancedTemporalFlowGNN(nn.Module):
    """
    Enhanced temporal GNN with flattened sequence input and explicit aggregation.
    
    Input: Sequence flattened per node: x_i = [u_i[t1:t6], v_i[t1:t6], p_i[t1:t6], ...]
    Output: Next time step: y_i = [u_i[t7], v_i[t7], p_i[t7], ...]
    """
    
    def __init__(self, fields_per_timestep: int = 4, sequence_length: int = 6,
                 hidden_dim: int = 64, num_layers: int = 3,
                 temporal_layers: int = 2, dropout: float = 0.1,
                 use_explicit_aggregation: bool = True):
        """
        Initialize Enhanced Temporal FlowGNN.
        
        Args:
            fields_per_timestep: Number of fields per time step (e.g., u, v, p, ... = 4)
            sequence_length: Number of input time steps (default: 6)
            hidden_dim: Hidden layer dimension
            num_layers: Number of GNN layers
            temporal_layers: Number of LSTM layers
            dropout: Dropout rate
            use_explicit_aggregation: If True, explicitly aggregate neighbor features
        """
        super(EnhancedTemporalFlowGNN, self).__init__()
        
        self.fields_per_timestep = fields_per_timestep
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.output_dim = fields_per_timestep  # Output is single time step
        
        # Input dimension: flattened sequence (fields * time_steps)
        input_dim = fields_per_timestep * sequence_length
        
        # Spatial GNN with explicit aggregation
        self.spatial_gnn = EnhancedFlowGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,  # Output hidden features for LSTM
            num_layers=num_layers,
            dropout=dropout,
            use_explicit_aggregation=use_explicit_aggregation
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
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_dim)
        )
        
        # Initialize for small increments
        nn.init.xavier_uniform_(self.output_proj[-1].weight, gain=0.1)
        nn.init.zeros_(self.output_proj[-1].bias)
    
    def forward(self, x_seq, edge_index, batch=None):
        """
        Forward pass for flattened sequence input.
        
        Args:
            x_seq: Flattened sequence features [num_nodes, fields * time_steps]
                   Format: [u[t1:t6], v[t1:t6], p[t1:t6], ...]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch vector (optional)
        
        Returns:
            Output [num_nodes, fields] for next time step
        """
        num_nodes = x_seq.shape[0]
        
        # Process through spatial GNN (handles neighbor aggregation)
        spatial_out = self.spatial_gnn(x_seq, edge_index, batch)
        
        # Reshape for LSTM: [num_nodes, 1, hidden_dim]
        spatial_out = spatial_out.unsqueeze(1)
        
        # Apply temporal LSTM
        lstm_out, _ = self.temporal_lstm(spatial_out)
        
        # Reshape: [num_nodes, hidden_dim]
        lstm_out = lstm_out.squeeze(1)
        
        # Output projection - predict increment
        delta = self.output_proj(lstm_out)
        
        # Add residual: extract last time step from input
        # Input format: [u[t1:t6], v[t1:t6], p[t1:t6], ...]
        # Extract last time step: [u[t6], v[t6], p[t6], ...]
        last_timestep = x_seq[:, -self.fields_per_timestep:]
        
        if last_timestep.shape == delta.shape:
            output = last_timestep + delta
        else:
            output = delta
        
        return output

