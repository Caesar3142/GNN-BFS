# Fluid Simulation Using Graph Neural Network (GNN)
## Method & Equation Summary

====================================================

## 1. Governing Physics (Reference Baseline)

**Traditional CFD solves Navier–Stokes equations:**

```
∂u/∂t + (u · ∇)u = −(1/ρ)∇p + ν∇²u + f
∇ · u = 0
```

- High accuracy but computationally expensive
- Used here **ONLY** to generate training data

---

## 2. Graph Representation of Fluid Domain

- Computational domain → Graph **G = (V, E)**
  - **V**: fluid nodes
  - **E**: edges representing neighborhood relations

- Each node stores physical states over time

---

## 3. Adaptive Neighbor Sampling (Inductive Bias)

**Neighbor set for node i:**

```
ne(i) = { j | (x_i − x_j)² + (y_i − y_j)² ≤ r }
```

- Sampling radius adapts to local gradient magnitude
- **Purpose:**
  - Capture multi-scale flow physics
  - Preserve sharp gradients (near walls, wakes)

---

## 4. Node Feature Definition

**Input feature at node i (6 previous time steps):**

```
x_i^(t-1:t-6) = [ c_i, p_i, u_i, v_i ]_(t-1:t-6)
```

**Output feature (next time step):**

```
y_i^t = [ c_i, p_i, u_i, v_i ]_t
```

**where:**
- **c**: concentration
- **p**: pressure
- **u**: x-velocity
- **v**: y-velocity

---

## 5. Graph-Based Time Evolution Model

**Learned simulator:**

```
y_i^t = Σ_{j∈ne(i)} f_ω( x_i^(t-1:t-6), x_j^(t-1:t-6), e_ij )
```

**where:**
- **f_ω**: trainable GNN function
- **e_ij**: edge attributes
- **ω**: network parameters

---

## 6. Graph Attention Mechanism (GAT)

**Linear projections:**

```
Q_i = W_q h_i
K_j = W_k h_j
V_j = W_v h_j
```

**Attention coefficient:**

```
α_ij = exp( σ(Q_i · K_j / √d) ) / Σ_{k∈ne(i)} exp( σ(Q_i · K_k / √d) )
```

**where:**
- **σ**: LeakyReLU
- **d**: dimension of Q and K

**Aggregation:**

```
h'_i = Σ_{j∈ne(i)} α_ij V_j
```

---

## 7. Multi-Head Attention

**K attention heads:**

```
h'_i = (1/K) Σ_{k=1..K} Σ_{j∈ne(i)} α_ij^(k) W^(k) h_j
```

**Purpose:**
- Learn different physical couplings
- Handle pressure–velocity nonlinearity

---

## 8. Deep Network with Residual Connections

- Single-head + multi-head GAT layers
- **Residual update:**

```
y^l = x^l + F(x^l, W^l)
x^(l+1) = σ(y^l)
```

**Enables:**
- Stable deep training
- Large receptive field
- Multi-hop physics propagation

---

## 9. Loss Function

**Mean Squared Error (MSE):**

```
L = || y_pred − y_CFD ||_2^2
```

**Evaluated using L2 relative error:**

```
L2_error = ||φ_pred − φ_true||_2 / ||φ_true||_2
```

**Note:** In our implementation, we also use:
- Field-weighted loss (separate weights for velocity and pressure)
- Relative error component for scale-aware learning
- Pressure-specific relative error for inlet pressure accuracy

---

## 10. Super-Resolution via Graph Aggregation

- Train on low-resolution nodes
- Apply model on high-resolution graph
- **Justification:**
  - Low-frequency flow dominates physics
  - High-frequency details reconstructed by GNN

---

## 11. Key Outcomes

- **Accuracy:** Comparable to CFD
- **Speedup:** 100–1000× vs traditional solvers
- **Generalizes across:**
  - Reynolds number
  - Boundary conditions
  - 2D and 3D flows

---

## Implementation Notes

### Current Implementation Features

1. **Adaptive Spatial Sampling**: Implemented in `graph_constructor.py`
   - Formula: `ne[i] = {j | (x_j - x_i)² + (y_j - y_i)² < r²}`
   - Auto-computes radius from mesh topology

2. **Graph Attention**: Custom attention mechanism in `attention_layers.py`
   - Single-head and multi-head attention
   - Aggregated query from neighbors

3. **Residual Connections**: Implemented in `gnn_model.py`
   - Predicts increments: `output = input + delta`
   - Preserves scale of input values

4. **Field-Specific Normalization**: In `normalization.py`
   - Separate normalization for velocity and pressure
   - Better handling of different scales

5. **Field-Weighted Loss**: In `train.py`
   - Separate loss computation for velocity and pressure
   - Pressure-specific relative error component

### Model Architecture

- **FlowGNN**: Static GNN for single time-step prediction
- **TemporalFlowGNN**: Sequence-to-sequence model with LSTM
- **EnhancedFlowGNN**: Custom attention with explicit neighbor aggregation

### Training Features

- Residual connections for scale preservation
- Gradient clipping for stability
- Learning rate scheduling
- Field-weighted loss with pressure emphasis
- Pressure-specific relative error loss

---

## References

For detailed theoretical background, see:
- [METHODOLOGY.md](METHODOLOGY.md) - Extended methodology discussion
- [ATTENTION_ARCHITECTURE.md](ATTENTION_ARCHITECTURE.md) - Detailed attention mechanism
- [FEATURES.md](FEATURES.md) - Feature overview

---

*Last updated: Based on GNN-BFS implementation*

