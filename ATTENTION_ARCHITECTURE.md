# Custom Attention Architecture

This document describes the custom attention mechanism implemented based on the cylinder project methodology.

## Architecture Overview

The network uses a combination of single-headed and multi-headed attention layers with specific residual connections to capture complex fluid dynamics relationships.

## Attention Mechanism

### Query, Key, Value Computation

For each node `i`:
- **Query (aggregated from neighbors)**: `q_i = sum(h[j] @ W_q for j in neighbors[i])`
- **Key (from current node)**: `k_i = h[i] @ W_k`
- **Value (from current node)**: `v_i = h[i] @ W_v`

### Attention Weight Calculation

The attention coefficient `α_ij` from node `i` to neighbor `j` is computed as:

```
score_ij = s(σ(k_i), σ(q_j)) / sqrt(d)
α_ij = softmax(score_ij for all j in neighbors[i])
```

Where:
- `s` is the dot product
- `σ` is the LeakyReLU activation function
- `d` is the dimension of query and key vectors
- `q_j` is the aggregated query from neighbors of node `j`

### Single-Head Attention Output

```
h_i' = σ(sum(α_ij * (h[j] @ W_l) for j in neighbors[i]))
```

### Multi-Head Attention Output

With `K=4` heads, each with K and Q dimension = 1:

```
h_i' = σ((1/K) * sum(sum(α[k]_ij * (h[j] @ W_l) for j in neighbors[i]) for k in range(K)))
```

## Network Architecture

### Layer Structure

1. **First Layer**: Single-headed attention
2. **Subsequent Layers**: Alternating multi-headed and single-headed attention
   - Layer 1: Single-head
   - Layer 2: Multi-head (4 heads)
   - Layer 3: Single-head
   - Layer 4: Multi-head (4 heads)
   - ...

### Residual Connections

Residual connections follow the pattern:
- **First layer**: `y_0 = x_0 + F(x_0, W_0)`
- **Every 2 layers after first**: `y_l = x_l + F(x_l, W_l)`
- **Between multi-head and single-head**: Residual connection

Formula:
```
y_l = h(x_l) + F(x_l, W_l)
x_{l+1} = σ(y_l)
```

This allows information from shallow layers to propagate to deep layers:
```
x_L = σ(x_l + sum(F(x_i, W_i) for i in range(l, L)))
```

## Activation Function

All layers use **LeakyReLU** activation (σ) with negative slope = 0.2.

## Usage

To use the custom attention architecture:

```bash
python train.py \
    --data_dir BFS-OpenFOAM-data \
    --epochs 100 \
    --model_type temporal \
    --sequence_length 6 \
    --use_enhanced_model \
    --use_flattened_input \
    --include_coordinates \
    --adaptive_sampling \
    --layer_type attention \
    --num_layers 4 \
    --hidden_dim 128
```

## Key Features

1. **Aggregated Query**: Query vector is aggregated from neighbors, not computed from current node
2. **Multi-Scale Information**: Deep network (k layers) captures k-order neighbor information
3. **Residual Connections**: Preserve shallow network information in deep layers
4. **Nonlinear Coupling**: Attention coefficients characterize nonlinear relationships between pressure, velocity, and other coupled physical quantities

## Implementation Details

- **SingleHeadAttention**: Implements single-headed attention with aggregated query
- **MultiHeadAttention**: Implements 4-head attention with K and Q dimension = 1
- **EnhancedFlowGNN**: Main model with alternating attention layers and residual connections

