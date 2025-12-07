# GNN-BFS: Method-Based Implementation

This directory contains an implementation following the **exact method specification** provided in [METHOD_SUMMARY.md](METHOD_SUMMARY.md).

## Overview

This implementation strictly follows the method specification:
- **Features**: [c, p, u, v] where c=concentration, p=pressure, u=x-velocity, v=y-velocity
- **Input**: 6 previous time steps flattened: x_i^(t-1:t-6) = [c_i, p_i, u_i, v_i]_(t-1:t-6)
- **Output**: Next time step: y_i^t = [c_i, p_i, u_i, v_i]_t
- **Loss**: MSE only: L = ||y_pred - y_CFD||_2^2
- **Evaluation**: L2 relative error: ||φ_pred - φ_true||_2 / ||φ_true||_2

## Files

### Core Implementation
- `graph_constructor_method.py`: Graph construction with [c, p, u, v] features
- `attention_method.py`: Exact GAT formula implementation
- `gnn_model_method.py`: Model with 6 time steps and residual connections
- `dataset_method.py`: Dataset with fixed 6 time steps input
- `train_method.py`: Training script with MSE loss
- `predict_method.py`: Prediction script with L2 error evaluation

## Installation

Same as main project. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the method-based model:

```bash
python train_method.py \
    --data_dir BFS-OpenFOAM-data \
    --epochs 100 \
    --batch_size 4 \
    --hidden_dim 64 \
    --num_layers 4 \
    --lr 0.001 \
    --adaptive_sampling \
    --normalize \
    --normalize_per_field
```

**Key Arguments:**
- `--data_dir`: Path to OpenFOAM data directory
- `--epochs`: Number of training epochs
- `--adaptive_sampling`: Use adaptive neighbor sampling (recommended)
- `--normalize`: Normalize features
- `--normalize_per_field`: Normalize each field separately
- `--include_concentration`: Include concentration field if available (defaults to zeros)

### Prediction

Make predictions:

```bash
python predict_method.py \
    --checkpoint checkpoints_method/best_model.pt \
    --data_dir BFS-OpenFOAM-data \
    --time_dir 0.096
```

## Method Specification Compliance

### ✅ Node Features
- **Format**: [c, p, u, v] = 4 features per time step
- **Concentration (c)**: Defaults to zeros if not in OpenFOAM data
- **Pressure (p)**: Scalar field from OpenFOAM
- **x-Velocity (u)**: First component of velocity vector
- **y-Velocity (v)**: Second component of velocity vector

### ✅ Input Format
- **6 time steps**: Fixed sequence length (t-6 to t-1)
- **Flattened**: [c, p, u, v]_(t-6), ..., [c, p, u, v]_(t-1) = 24 features

### ✅ Adaptive Neighbor Sampling
- **Formula**: ne(i) = { j | (x_i − x_j)² + (y_i − y_j)² ≤ r }
- **Auto-radius**: Computed from mesh topology
- **Purpose**: Capture multi-scale flow physics

### ✅ Graph Attention Mechanism
- **Q, K, V projections**: Q_i = W_q h_i, K_j = W_k h_j, V_j = W_v h_j
- **Attention**: α_ij = exp(σ(Q_i · K_j / √d)) / Σ exp(σ(Q_i · K_k / √d))
- **Aggregation**: h'_i = Σ_{j∈ne(i)} α_ij V_j
- **Activation**: σ = LeakyReLU

### ✅ Multi-Head Attention
- **Formula**: h'_i = (1/K) Σ_{k=1..K} Σ_{j∈ne(i)} α_ij^(k) W^(k) h_j
- **Heads**: 4 heads (K=4)
- **Purpose**: Learn different physical couplings

### ✅ Residual Connections
- **Formula**: y^l = x^l + F(x^l, W^l), x^(l+1) = σ(y^l)
- **Pattern**: First layer, then every 2 layers
- **Purpose**: Stable deep training, large receptive field

### ✅ Loss Function
- **Training**: L = ||y_pred - y_CFD||_2^2 (MSE)
- **Evaluation**: L2_error = ||φ_pred - φ_true||_2 / ||φ_true||_2

## Model Architecture

### FlowGNNMethod
- **Input**: 24 features (4 fields × 6 time steps)
- **Output**: 4 features (c, p, u, v)
- **Layers**: Alternating single-head and multi-head GAT
- **Residual**: Applied at first layer and every 2 layers

### TemporalFlowGNNMethod
- **Spatial**: FlowGNNMethod processes spatial relationships
- **Temporal**: LSTM models time dependencies
- **Output**: Next time step prediction

## Differences from Original Implementation

| Aspect | Original | Method-Based |
|--------|----------|--------------|
| Features | [u, v, w, p] | [c, p, u, v] |
| Time steps | Variable (3-5) | Fixed (6) |
| Attention | Custom (aggregated query) | Standard GAT formula |
| Loss | Huber + relative error | MSE only |
| Evaluation | Multiple metrics | L2 relative error |

## Notes

1. **Concentration Field**: If your OpenFOAM data doesn't have a concentration field, it will default to zeros. This is acceptable for flow-only simulations.

2. **6 Time Steps**: The method requires exactly 6 previous time steps. Ensure your data has at least 7 time directories (6 for input + 1 for target).

3. **Feature Order**: Features are always in order [c, p, u, v]. Make sure this matches your data.

4. **Adaptive Sampling**: Recommended for better gradient capture, especially near walls and obstacles.

## Troubleshooting

**Error: "Need at least 6 previous time steps"**
- Ensure you have at least 7 time directories in your data
- Check that time directories are properly sorted

**Concentration is all zeros**
- This is expected if concentration field is not in OpenFOAM data
- Use `--include_concentration` only if you have a concentration field

**Model not learning**
- Try `--normalize` and `--normalize_per_field`
- Increase `--num_layers` or `--hidden_dim`
- Use `--adaptive_sampling` for better gradient capture

## References

- [METHOD_SUMMARY.md](METHOD_SUMMARY.md) - Method specification
- [METHODOLOGY.md](METHODOLOGY.md) - Extended methodology
- [README.md](README.md) - Main project documentation

