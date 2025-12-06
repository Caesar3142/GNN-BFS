# Features Overview

This document provides an overview of all features implemented in the GNN-BFS project.

## Core Features

### 1. OpenFOAM Data Parsing
- **Mesh parsing**: Extracts points, faces, owner, and neighbour from OpenFOAM mesh files
- **Field parsing**: Handles both uniform and nonuniform field formats
- **Time directory handling**: Automatically discovers and processes time directories
- **Robust parsing**: Handles various OpenFOAM file formats and edge cases

### 2. Graph Construction
- **Mesh-based connectivity**: Builds edges from OpenFOAM face connectivity
- **Adaptive spatial sampling**: Optional spatial radius-based neighbor sampling
- **Bidirectional edges**: Ensures graph symmetry for message passing
- **Cell center computation**: Accurate centroids from mesh geometry

### 3. Model Architectures

#### Standard Models
- **FlowGNN**: Static GNN for single time-step prediction
- **TemporalFlowGNN**: Sequence-to-sequence model with LSTM

#### Enhanced Models
- **EnhancedFlowGNN**: Custom attention architecture with explicit neighbor aggregation
- **EnhancedTemporalFlowGNN**: Temporal model with flattened sequence input

### 4. Custom Attention Mechanism
- **Single-headed attention**: Aggregated query from neighbors, key from current node
- **Multi-headed attention**: 4 heads with K and Q dimension = 1
- **Residual connections**: Preserves information through deep networks
- **LeakyReLU activation**: Consistent activation throughout

### 5. Training Features
- **Residual connections**: Predicts increments (delta) to preserve scale
- **Gradient clipping**: Prevents exploding gradients
- **Learning rate scheduling**: Adaptive learning rate reduction
- **Checkpointing**: Saves best model and training state
- **Training curves**: Visualizes loss over epochs

### 6. Visualization

#### Contour Plots
- **Vertical layout**: Stacked plots for larger visualization
- **Domain masking**: Delaunay triangulation preserves sharp boundaries
- **Consistent scales**: Same colorbar for OpenFOAM and prediction
- **Error visualization**: Separate error plots with units

#### Field Comparison
- **Scatter plots**: Ground truth vs prediction
- **Error distributions**: Histograms and statistics
- **R² scores**: Quantitative accuracy metrics
- **Component-wise analysis**: Separate Ux, Uy, Uz analysis

## Advanced Features

### Adaptive Spatial Sampling
- **Formula**: `ne[i] = {j | (x_j - x_i)² + (y_j - y_i)² < r²}`
- **Auto-radius**: Computes optimal radius from mesh topology
- **Multi-scale information**: Captures gradient information at multiple scales
- **High-gradient regions**: Better handling near walls and obstacles

### Coordinate Embedding
- **Cell center coordinates**: Optional embedding in node features
- **Normalization**: Coordinates normalized to [0, 1] range
- **Implicit geometry**: Helps model learn spatial relationships

### Flattened Sequence Input
- **Format**: `x_i = [u_i[t1:t6], v_i[t1:t6], p_i[t1:t6], ...]`
- **Single vector**: All time steps concatenated per node
- **Output**: Single time step prediction `y_i = [u_i[t7], v_i[t7], p_i[t7], ...]`

### Explicit Neighbor Aggregation
- **Function**: `y_i = f(x_i[t1:t6], sum(x_j[t1:t6] for j in neighbors[i]), e_ij, omega)`
- **Explicit computation**: Aggregates neighbor features before model processing
- **Better coupling**: Captures nonlinear relationships in Navier-Stokes equations

## Technical Details

### Data Processing
- **Field extraction**: Velocity (3D) and pressure (1D) by default
- **Normalization**: Optional feature normalization
- **Sequence handling**: Temporal sequences with configurable length
- **Batch processing**: Efficient batching with PyTorch Geometric

### Model Features
- **Scale preservation**: Residual connections maintain input scale
- **Deep networks**: k layers capture k-order neighbor information
- **Dropout**: Regularization for better generalization
- **Batch normalization**: Stabilizes training

### Visualization Features
- **Interpolation**: Grid-based interpolation for smooth contours
- **Domain masking**: Preserves actual mesh boundaries
- **Color mapping**: Consistent colormaps across plots
- **Error units**: Proper units displayed (m/s for velocity, m²/s² for pressure)

## Usage Examples

### Basic Training
```bash
python train.py --data_dir BFS-OpenFOAM-data --epochs 50 --model_type temporal
```

### Advanced Training
```bash
python train.py \
    --data_dir BFS-OpenFOAM-data \
    --epochs 100 \
    --use_enhanced_model \
    --use_flattened_input \
    --include_coordinates \
    --adaptive_sampling \
    --layer_type attention
```

### Visualization
```bash
# Contour plots
python plot_contours.py --checkpoint checkpoints/best_model.pt --data_dir BFS-OpenFOAM-data

# Field comparison
python plot_fields.py --checkpoint checkpoints/best_model.pt --data_dir BFS-OpenFOAM-data
```

## Performance Considerations

- **Memory**: Adaptive sampling increases edge count (~2x)
- **Training time**: Enhanced model with attention is slower but more accurate
- **Visualization**: Higher resolution grids (300x300) provide better detail
- **Checkpointing**: Saves disk space by keeping only best model

## Future Enhancements

Potential improvements:
- Gradient-based adaptive sampling
- Multi-scale graph construction
- Attention visualization
- Real-time prediction
- Uncertainty quantification

