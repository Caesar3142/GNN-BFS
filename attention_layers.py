"""
Custom attention layers based on cylinder project methodology.
Implements single-headed and multi-headed attention with specific architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter


class SingleHeadAttention(nn.Module):
    """
    Single-headed attention layer with aggregated query from neighbors.
    
    Attention mechanism:
    - q = sum(h[j] @ W_q for j in neighbors[i])  # Aggregated query from neighbors
    - k_i = h[i] @ W_k  # Key from current node
    - v_i = h[i] @ W_v  # Value from current node
    - alpha_ij = softmax(s(σ(k_i), σ(q_j)) / sqrt(d))
    """
    
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        """
        Initialize single-headed attention layer.
        
        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension
            dropout: Dropout rate
        """
        super(SingleHeadAttention, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        
        # Linear transformations for Q, K, V
        self.W_q = nn.Linear(in_dim, out_dim, bias=False)  # Query (from neighbors)
        self.W_k = nn.Linear(in_dim, out_dim, bias=False)  # Key (from current node)
        self.W_v = nn.Linear(in_dim, out_dim, bias=False)  # Value (from current node)
        
        # LeakyReLU activation (σ)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        
        # Output projection
        self.W_l = nn.Linear(in_dim, out_dim)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_l.weight)
    
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge connectivity [2, num_edges]
        
        Returns:
            Output features [num_nodes, out_dim]
        """
        num_nodes = x.size(0)
        row, col = edge_index  # row: source nodes, col: target nodes
        
        # Compute Q, K, V
        # q: aggregated query from neighbors
        # For each node i, q_i = sum(h[j] @ W_q for j in neighbors[i])
        h_q = self.W_q(x)  # [num_nodes, out_dim]
        h_k = self.W_k(x)  # [num_nodes, out_dim]
        h_v = self.W_v(x)  # [num_nodes, out_dim]
        
        # Aggregate query from neighbors: q_i = sum(h[j] @ W_q for j in neighbors[i])
        # For node i, we need to sum queries from all its neighbors
        # edge_index: row[i] -> col[i] means col[i] is neighbor of row[i]
        # So for node i, neighbors are col[row == i]
        q_aggregated = scatter(h_q[col], row, dim=0, dim_size=num_nodes, reduce='sum')
        
        # Apply LeakyReLU activation
        q_activated = self.leaky_relu(q_aggregated)  # [num_nodes, out_dim]
        k_activated = self.leaky_relu(h_k)  # [num_nodes, out_dim]
        
        # Compute attention scores
        # For each edge (i, j), score = s(σ(k_i), σ(q_j)) / sqrt(d)
        # where q_j is the aggregated query from neighbors of j
        # Actually, we need: for node i, attention to neighbor j
        # score_ij = dot(σ(k_i), σ(q_j)) / sqrt(d)
        # where q_j = sum(h[k] @ W_q for k in neighbors[j])
        
        # For each edge (i, j), compute attention score
        k_i = k_activated[row]  # [num_edges, out_dim] - key of source node
        q_j = q_aggregated[col]  # [num_edges, out_dim] - aggregated query of target node
        
        # Dot product attention: s(σ(k_i), σ(q_j))
        scores = (k_i * q_j).sum(dim=1, keepdim=True)  # [num_edges, 1]
        scores = scores / (self.out_dim ** 0.5)  # Scale by sqrt(d)
        
        # Apply softmax to get attention coefficients
        # alpha_ij = softmax(scores) for each node i
        alpha = scatter_softmax(scores.squeeze(), row, num_nodes)  # [num_edges]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Aggregate values with attention weights
        # h_i' = σ(sum(alpha_ij * (h[j] @ W_l) for j in neighbors[i]))
        h_proj = self.W_l(x)  # [num_nodes, out_dim]
        h_j = h_proj[col]  # [num_edges, out_dim] - features of neighbor nodes
        
        # Weighted aggregation
        alpha_expanded = alpha.unsqueeze(1)  # [num_edges, 1]
        weighted_features = alpha_expanded * h_j  # [num_edges, out_dim]
        
        # Aggregate: sum over neighbors
        h_prime = scatter(weighted_features, row, dim=0, dim_size=num_nodes, reduce='sum')
        
        # Apply LeakyReLU activation
        output = self.leaky_relu(h_prime)
        
        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-headed attention layer with 4 heads.
    K and Q representation subspace dimension = 1.
    """
    
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Initialize multi-headed attention layer.
        
        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension
            num_heads: Number of attention heads (default: 4)
            dropout: Dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # For each head, K and Q dimension = 1
        # So we have num_heads attention mechanisms, each with dim=1
        self.head_dim = 1
        
        # Linear transformations for each head
        self.W_q_heads = nn.ModuleList([
            nn.Linear(in_dim, self.head_dim, bias=False) for _ in range(num_heads)
        ])
        self.W_k_heads = nn.ModuleList([
            nn.Linear(in_dim, self.head_dim, bias=False) for _ in range(num_heads)
        ])
        self.W_v_heads = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_heads)
        ])
        
        # Output projection
        self.W_l = nn.Linear(in_dim, out_dim)
        
        # LeakyReLU activation
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        for W_q, W_k, W_v in zip(self.W_q_heads, self.W_k_heads, self.W_v_heads):
            nn.init.xavier_uniform_(W_q.weight)
            nn.init.xavier_uniform_(W_k.weight)
            nn.init.xavier_uniform_(W_v.weight)
        nn.init.xavier_uniform_(self.W_l.weight)
    
    def forward(self, x, edge_index):
        """
        Forward pass for multi-headed attention.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge connectivity [2, num_edges]
        
        Returns:
            Output features [num_nodes, out_dim]
        """
        num_nodes = x.size(0)
        row, col = edge_index
        
        # Process each head
        head_outputs = []
        
        for head_idx in range(self.num_heads):
            W_q = self.W_q_heads[head_idx]
            W_k = self.W_k_heads[head_idx]
            W_v = self.W_v_heads[head_idx]
            
            # Compute Q, K, V for this head
            h_q = W_q(x)  # [num_nodes, 1]
            h_k = W_k(x)  # [num_nodes, 1]
            h_v = W_v(x)  # [num_nodes, out_dim]
            
            # Aggregate query from neighbors
            q_aggregated = scatter(h_q[col], row, dim=0, dim_size=num_nodes, reduce='sum')
            
            # Apply LeakyReLU
            q_activated = self.leaky_relu(q_aggregated)  # [num_nodes, 1]
            k_activated = self.leaky_relu(h_k)  # [num_nodes, 1]
            
            # Compute attention scores
            k_i = k_activated[row]  # [num_edges, 1]
            q_j = q_aggregated[col]  # [num_edges, 1]
            
            # Dot product (since dim=1, this is just multiplication)
            scores = (k_i * q_j).squeeze()  # [num_edges]
            scores = scores / (self.head_dim ** 0.5)  # Scale by sqrt(d)
            
            # Softmax
            alpha = scatter_softmax(scores, row, num_nodes)  # [num_edges]
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            
            # Aggregate values
            h_proj = self.W_l(x)  # [num_nodes, out_dim]
            h_j = h_proj[col]  # [num_edges, out_dim]
            
            alpha_expanded = alpha.unsqueeze(1)  # [num_edges, 1]
            weighted_features = alpha_expanded * h_j  # [num_edges, out_dim]
            
            h_head = scatter(weighted_features, row, dim=0, dim_size=num_nodes, reduce='sum')
            head_outputs.append(h_head)
        
        # Average over heads: (1/K) * sum(head_k for k in range(K))
        h_prime = torch.stack(head_outputs, dim=0).mean(dim=0)  # [num_nodes, out_dim]
        
        # Apply LeakyReLU activation
        output = self.leaky_relu(h_prime)
        
        return output


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

