# Methodology and Theory

This document describes the theoretical background and methodology behind the GNN-BFS project for flow field prediction using Graph Neural Networks.

## Overview

The project implements Graph Neural Networks (GNNs) for learning and predicting fluid flow fields from OpenFOAM simulation data. The approach leverages the natural graph structure of computational fluid dynamics (CFD) meshes to model complex physical relationships in Navier-Stokes equations.

## Graph Neural Networks for CFD

### Why GNNs for Fluid Dynamics?

Navier-Stokes equations describe fluid flow with strong coupling relationships between physical quantities (pressure, velocity, density). The universal approximation theorem states that sufficiently large and deep networks can approximate any function. Graph neural networks have:

1. **High parameter complexity**: Can model complex nonlinear relationships
2. **Natural similarity to CFD**: Both use aggregation mechanisms to propagate information
3. **Mesh structure**: CFD meshes are naturally represented as graphs

### Graph Representation of CFD Mesh

In OpenFOAM, the computational domain is discretized into:
- **Cells**: Volumetric elements (nodes in the graph)
- **Faces**: Interfaces between cells (edges in the graph)
- **Points**: Vertices of cells (used for geometry)

The graph structure is built from:
- **Nodes**: Cell centers with features (velocity, pressure, etc.)
- **Edges**: Connections between adjacent cells via faces
- **Features**: Physical quantities at each cell

## Model Architecture

### Standard Models

#### FlowGNN (Static)
- **Input**: Node features at single time step `x_i = [u_i, v_i, w_i, p_i]`
- **Architecture**: Multi-layer GCN/GAT/GIN with residual connections
- **Output**: Predicted features at next time step `y_i = [u_i[t+1], v_i[t+1], w_i[t+1], p_i[t+1]]`
- **Function**: `y_i = f(x_i, {x_j for j in neighbors[i]}, e_ij, ω)`

#### TemporalFlowGNN
- **Input**: Sequence of node features `x_i[t1:t6]`
- **Architecture**: Spatial GNN + Temporal LSTM
- **Output**: Predicted sequence `y_i[t7]`
- **Temporal modeling**: LSTM captures time dependencies

### Enhanced Model with Custom Attention

The enhanced model uses a custom attention mechanism to better capture nonlinear coupling relationships.

#### Input Format
- **Flattened sequence**: `x_i = [u_i[t1:t6], v_i[t1:t6], p_i[t1:t6], ...]`
- All 6 time steps concatenated per field per node
- **Output**: Single time step `y_i = [u_i[t7], v_i[t7], p_i[t7], ...]`

#### Function Form
```
y_i = f(
    x_i[t1:t6],
    sum(x_j[t1:t6] for j in neighbors[i]),
    e_ij,
    ω
)
```

Where:
- `x_i[t1:t6]`: Input sequence for node i
- `sum(x_j[t1:t6] for j in neighbors[i])`: Explicit neighbor aggregation
- `e_ij`: Edge connection between nodes i and j
- `ω`: Learnable parameters

## Attention Mechanism

### Single-Headed Attention

The attention mechanism establishes weight matrices between computing nodes and neighboring nodes:

**Query, Key, Value Computation:**
- **Query (aggregated from neighbors)**: `q_i = sum(h[j] @ W_q for j in neighbors[i])`
- **Key (from current node)**: `k_i = h[i] @ W_k`
- **Value (from current node)**: `v_i = h[i] @ W_v`

**Attention Weight:**
```
score_ij = s(σ(k_i), σ(q_j)) / sqrt(d)
α_ij = softmax(score_ij for all j in neighbors[i])
```

Where:
- `s`: Dot product
- `σ`: LeakyReLU activation function
- `d`: Dimension of query and key vectors
- `α_ij`: Attention coefficient

**Output:**
```
h_i' = σ(sum(α_ij * (h[j] @ W_l) for j in neighbors[i]))
```

### Multi-Headed Attention

The multi-headed attention mechanism uses 4 heads, each with K and Q dimension = 1:

```
h_i' = σ((1/K) * sum(sum(α[k]_ij * (h[j] @ W_l) for j in neighbors[i]) for k in range(K)))
```

### Why Attention for Navier-Stokes?

1. **Nonlinear coupling**: Pressure and velocity are nonlinearly coupled in Navier-Stokes equations
2. **Attention coefficients**: Characterize nonlinear relationships between node features
3. **Weight allocation**: Attention weights allocate importance between nodes
4. **Multi-scale information**: Multiple heads capture different relationship aspects

## Adaptive Spatial Sampling

### Problem Statement

In CFD, grids are refined near walls and obstacles where gradient changes are significant. Selecting only first-order neighbors or a fixed number of neighbors makes gradient differences too averaged, causing the neural network to miss drastic gradient changes in high-gradient regions.

### Solution: Adaptive Sampling

**Formula:**
```
ne[i] = {j | (x_j - x_i)² + (y_j - y_i)² < r²}
```

**Benefits:**
1. **Multi-scale gradient information**: Samples many neighbors in high-gradient regions
2. **Preserves details**: Describes details while maintaining overall information
3. **Prevents jitter**: Reduces oscillations in flow fields with large gradient changes
4. **Better perception**: Neural network can perceive drastic gradient changes

### Comparison

- **Fixed neighbor sampling**: Average gradient difference, misses high-gradient regions
- **Adaptive sampling**: Captures multi-scale information, better for complex flows

## Residual Connections

### Problem: Information Loss in Deep Networks

As physical information propagates through deep networks, shallow network information is gradually lost. To transfer shallow network information to deep networks while maintaining network complexity, residual connections are added.

### Residual Connection Formula

```
y_l = h(x_l) + F(x_l, W_l)
x_{l+1} = σ(y_l)
```

Where:
- `x_l`: Input of layer l
- `y_l`: Output of layer l
- `F`: Attention layer function
- `h(x_l)`: Identity mapping (residual)
- `σ`: LeakyReLU activation function

### Information Propagation

With residual connections, information from shallow layer l to deep layer L:

```
x_L = σ(x_l + sum(F(x_i, W_i) for i in range(l, L)))
```

This ensures that:
- Shallow information is preserved
- Deep layers can learn corrections
- Network complexity is maintained

### Residual Pattern

- **First layer**: `y_0 = x_0 + F(x_0, W_0)`
- **Every 2 layers**: `y_l = x_l + F(x_l, W_l)`
- **Between attention types**: Residual between multi-head and single-head layers

## Scale Preservation

### Problem: Output Scale Mismatch

Initial models predicted values in wrong scale (e.g., pressure predicted as [-0.1, 0.1] instead of [15, 375]).

### Solution: Residual Connections for Increments

Instead of predicting absolute values:
```
output = model(input)  # Wrong scale
```

The model predicts increments:
```
delta = model(input)
output = input + delta  # Preserves scale
```

This ensures:
- **Scale preservation**: Outputs maintain input scale automatically
- **Small increments**: Model learns to predict changes, not absolute values
- **Proper initialization**: Output layers initialized to predict small increments

## Network Depth and Receptive Field

### k-Layer Network

A k-layer graph convolutional neural network can fuse information from k-order neighbor nodes. Therefore, deep networks have wider receptive fields and can capture farther physical information.

**Example (2-layer network):**
- Layer 1: Aggregates from 1st-order neighbors
- Layer 2: Aggregates from 2nd-order neighbors (neighbors of neighbors)
- Total receptive field: 2-hop neighborhood

### Information Aggregation

The following figure shows aggregation node information of a two-layer graph neural network:

```
Node i
  ↓ (Layer 1)
Neighbors of i
  ↓ (Layer 2)
Neighbors of neighbors of i
```

## Coordinate Embedding

### Implicit Geometry Information

Node coordinate information is implicitly embedded into the input vector. This helps the model learn spatial relationships and geometric features of the flow domain.

**Implementation:**
- Cell center coordinates `(x, y, z)` are normalized to [0, 1] range
- Added as additional features to node input
- Helps model understand spatial structure

## Training Methodology

### Loss Function

Mean Squared Error (MSE) between predicted and ground truth field values:

```
L = (1/N) * sum((y_pred - y_true)²)
```

### Optimization

- **Optimizer**: Adam
- **Learning rate**: Adaptive reduction with ReduceLROnPlateau
- **Gradient clipping**: Prevents exploding gradients (max_norm=1.0)
- **Initialization**: Output layers initialized for small increments

### Data Splitting

- **Train/Validation split**: 80/20 by default
- **Temporal sequences**: Input sequence of length 6, predict 1 step ahead
- **Batch processing**: Efficient batching with PyTorch Geometric

## Physical Interpretation

### Navier-Stokes Equations

The Navier-Stokes equations describe fluid motion:

```
∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + f
∇·u = 0
```

Where:
- `u`: Velocity vector
- `p`: Pressure
- `ρ`: Density
- `ν`: Kinematic viscosity
- `f`: Body forces

### GNN Approximation

The GNN approximates the solution operator:
- **Input**: Current state `u(t), p(t)`
- **Output**: Next state `u(t+Δt), p(t+Δt)`
- **Function**: Learned from data rather than solving PDEs directly

### Advantages

1. **Fast inference**: Once trained, predictions are much faster than CFD simulation
2. **Data-driven**: Learns from simulation data without explicit physics
3. **Generalization**: Can predict for similar flow conditions
4. **Uncertainty**: Can quantify prediction uncertainty

## Limitations and Future Work

### Current Limitations

1. **Training data**: Requires sufficient OpenFOAM simulation data
2. **Generalization**: May not generalize to very different flow conditions
3. **Physics**: Not explicitly enforcing physical constraints
4. **Temporal**: Limited to short-term predictions

### Future Enhancements

1. **Physics-informed**: Incorporate physical constraints in loss function
2. **Multi-scale**: Hierarchical graph construction
3. **Uncertainty**: Quantify prediction uncertainty
4. **Transfer learning**: Pre-train on multiple flow cases
5. **Real-time**: Optimize for real-time prediction

## References

- Graph Neural Networks for CFD: Natural combination due to aggregation mechanisms
- Universal Approximation Theorem: Sufficiently large networks can approximate any function
- Attention Mechanisms: Characterize nonlinear relationships in coupled systems
- Residual Connections: Preserve information in deep networks
- Adaptive Sampling: Capture multi-scale gradient information

## Summary

The GNN-BFS methodology combines:
1. **Graph structure**: Natural representation of CFD meshes
2. **Attention mechanism**: Captures nonlinear coupling relationships
3. **Adaptive sampling**: Handles high-gradient regions
4. **Residual connections**: Preserves information in deep networks
5. **Scale preservation**: Predicts increments to maintain proper scale
6. **Temporal modeling**: LSTM captures time dependencies

This approach enables fast, accurate flow field prediction while learning complex physical relationships from simulation data.

