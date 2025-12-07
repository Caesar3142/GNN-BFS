"""
Graph Attention Mechanism following exact method specification.

GAT formula:
- Q_i = W_q h_i
- K_j = W_k h_j  
- V_j = W_v h_j
- α_ij = exp(σ(Q_i · K_j / √d)) / Σ_{k∈ne(i)} exp(σ(Q_i · K_k / √d))
- h'_i = Σ_{j∈ne(i)} α_ij V_j

where σ is LeakyReLU.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter


class GATLayerMethod(nn.Module):
    """
    Graph Attention Layer following exact method specification.
    
    Attention coefficient:
    α_ij = exp(σ(Q_i · K_j / √d)) / Σ_{k∈ne(i)} exp(σ(Q_i · K_k / √d))
    
    Aggregation:
    h'_i = Σ_{j∈ne(i)} α_ij V_j
    """
    
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1, negative_slope: float = 0.2):
        """
        Initialize GAT layer.
        
        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension
            dropout: Dropout rate
            negative_slope: LeakyReLU negative slope
        """
        super(GATLayerMethod, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.negative_slope = negative_slope
        
        # Linear transformations: Q, K, V
        self.W_q = nn.Linear(in_dim, out_dim, bias=False)  # Query from current node
        self.W_k = nn.Linear(in_dim, out_dim, bias=False)  # Key from neighbor
        self.W_v = nn.Linear(in_dim, out_dim, bias=False)  # Value from neighbor
        
        # LeakyReLU activation (σ)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        
        # Scale factor: √d
        self.scale = (out_dim ** 0.5)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
    
    def forward(self, x, edge_index):
        """
        Forward pass following exact GAT formula.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge connectivity [2, num_edges]
        
        Returns:
            Output features [num_nodes, out_dim]
        """
        num_nodes = x.size(0)
        row, col = edge_index  # row: source nodes, col: target nodes
        
        # Linear projections
        Q = self.W_q(x)  # [num_nodes, out_dim] - Query from current node
        K = self.W_k(x)  # [num_nodes, out_dim] - Key from all nodes
        V = self.W_v(x)  # [num_nodes, out_dim] - Value from all nodes
        
        # For each edge (i, j):
        # - Q_i: query from source node i
        # - K_j: key from target node j (neighbor of i)
        # - V_j: value from target node j
        
        Q_i = Q[row]  # [num_edges, out_dim] - Query from source nodes
        K_j = K[col]  # [num_edges, out_dim] - Key from target nodes (neighbors)
        V_j = V[col]  # [num_edges, out_dim] - Value from target nodes (neighbors)
        
        # Apply LeakyReLU activation
        Q_i_activated = self.leaky_relu(Q_i)  # [num_edges, out_dim]
        K_j_activated = self.leaky_relu(K_j)  # [num_edges, out_dim]
        
        # Compute attention scores: Q_i · K_j / √d
        scores = (Q_i_activated * K_j_activated).sum(dim=1)  # [num_edges]
        scores = scores / self.scale  # Divide by √d
        
        # Apply softmax: α_ij = exp(score_ij) / Σ_{k∈ne(i)} exp(score_ik)
        alpha = scatter_softmax(scores, row, num_nodes)  # [num_edges]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Weighted aggregation: h'_i = Σ_{j∈ne(i)} α_ij V_j
        alpha_expanded = alpha.unsqueeze(1)  # [num_edges, 1]
        weighted_V = alpha_expanded * V_j  # [num_edges, out_dim]
        
        # Aggregate over neighbors
        h_prime = scatter(weighted_V, row, dim=0, dim_size=num_nodes, reduce='sum')
        
        return h_prime


class MultiHeadGATMethod(nn.Module):
    """
    Multi-Head Attention following exact method specification.
    
    Formula:
    h'_i = (1/K) Σ_{k=1..K} Σ_{j∈ne(i)} α_ij^(k) W^(k) h_j
    """
    
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, 
                 dropout: float = 0.1, negative_slope: float = 0.2):
        """
        Initialize multi-head GAT layer.
        
        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension
            num_heads: Number of attention heads (K)
            dropout: Dropout rate
            negative_slope: LeakyReLU negative slope
        """
        super(MultiHeadGATMethod, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Create K attention heads
        self.heads = nn.ModuleList([
            GATLayerMethod(in_dim, out_dim, dropout, negative_slope)
            for _ in range(num_heads)
        ])
        
        # Output projection for each head: W^(k)
        self.W_heads = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(num_heads)
        ])
    
    def forward(self, x, edge_index):
        """
        Forward pass for multi-head attention.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge connectivity [2, num_edges]
        
        Returns:
            Output features [num_nodes, out_dim]
        """
        num_nodes = x.size(0)
        row, col = edge_index
        
        head_outputs = []
        
        # Process each head
        for head_idx, (gat_head, W_head) in enumerate(zip(self.heads, self.W_heads)):
            num_nodes = x.size(0)
            row, col = edge_index
            
            # Compute Q, K for attention coefficients
            Q = gat_head.W_q(x)
            K = gat_head.W_k(x)
            
            Q_i = Q[row]
            K_j = K[col]
            
            Q_i_activated = gat_head.leaky_relu(Q_i)
            K_j_activated = gat_head.leaky_relu(K_j)
            
            # Attention scores: α_ij^(k)
            scores = (Q_i_activated * K_j_activated).sum(dim=1)
            scores = scores / gat_head.scale
            
            # Softmax: α_ij^(k)
            alpha = scatter_softmax(scores, row, num_nodes)
            alpha = F.dropout(alpha, p=gat_head.dropout, training=self.training)
            
            # Project features: W^(k) h_j
            h_proj = W_head(x)  # [num_nodes, out_dim]
            h_j = h_proj[col]  # [num_edges, out_dim]
            
            # Weighted aggregation: Σ_{j∈ne(i)} α_ij^(k) W^(k) h_j
            alpha_expanded = alpha.unsqueeze(1)  # [num_edges, 1]
            weighted_h = alpha_expanded * h_j  # [num_edges, out_dim]
            h_head = scatter(weighted_h, row, dim=0, dim_size=num_nodes, reduce='sum')
            
            head_outputs.append(h_head)
        
        # Average over heads: (1/K) Σ_{k=1..K} ...
        h_prime = torch.stack(head_outputs, dim=0).mean(dim=0)  # [num_nodes, out_dim]
        
        return h_prime
    


def scatter_softmax(src, index, num_nodes):
    """
    Compute softmax over neighbors for each node.
    
    Args:
        src: Source values [num_edges]
        index: Node indices [num_edges]
        num_nodes: Total number of nodes
    
    Returns:
        Softmax-normalized values [num_edges]
    """
    # Compute max for numerical stability
    max_val = scatter(src, index, dim=0, dim_size=num_nodes, reduce='max')
    max_val = max_val[index]  # [num_edges]
    
    # Compute exp(x - max)
    exp_scores = torch.exp(src - max_val)
    
    # Sum over neighbors
    sum_exp = scatter(exp_scores, index, dim=0, dim_size=num_nodes, reduce='sum')
    sum_exp = sum_exp[index]  # [num_edges]
    
    # Normalize
    alpha = exp_scores / (sum_exp + 1e-8)
    
    return alpha

