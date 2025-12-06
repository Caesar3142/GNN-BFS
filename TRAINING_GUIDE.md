# Training Guide

This guide explains how to train the GNN model for flow prediction.

## Quick Start

1. **Activate the virtual environment:**
```bash
source venv/bin/activate
```

2. **Run basic training:**
```bash
python train.py --data_dir BFS-OpenFOAM-data --epochs 50 --model_type temporal
```

## Basic Training Commands

### Temporal Model (Recommended)

The temporal model uses LSTM to capture time dependencies:

```bash
python train.py \
    --data_dir BFS-OpenFOAM-data \
    --epochs 50 \
    --batch_size 4 \
    --model_type temporal \
    --sequence_length 3 \
    --hidden_dim 64 \
    --num_layers 3 \
    --lr 0.001
```

### Static Model

For single time-step prediction:

```bash
python train.py \
    --data_dir BFS-OpenFOAM-data \
    --epochs 50 \
    --batch_size 4 \
    --model_type static \
    --hidden_dim 64 \
    --num_layers 3 \
    --lr 0.001
```

## Training Parameters

### Essential Parameters

- `--data_dir`: Path to OpenFOAM data directory (default: `BFS-OpenFOAM-data`)
- `--epochs`: Number of training epochs (default: 50, recommended: 50-200)
- `--batch_size`: Batch size (default: 4, adjust based on GPU memory)
- `--model_type`: `temporal` or `static` (default: `temporal`)
- `--lr`: Learning rate (default: 0.001)

### Model Architecture

- `--hidden_dim`: Hidden layer dimension (default: 64, try 128 or 256 for larger models)
- `--num_layers`: Number of GNN layers (default: 3, try 4-6 for deeper models)
- `--layer_type`: GNN layer type - `GCN`, `GAT`, or `GIN` (default: `GCN`)

### Temporal Model Specific

- `--sequence_length`: Input sequence length (default: 3, try 5-10)
- `--predict_ahead`: Steps ahead to predict (default: 1)

### Data Splitting

- `--train_split`: Train/validation split ratio (default: 0.8)

### Output

- `--save_dir`: Directory to save checkpoints (default: `checkpoints`)

## Recommended Training Configurations

### 1. Quick Test (Fast, Lower Accuracy)

```bash
python train.py \
    --data_dir BFS-OpenFOAM-data \
    --epochs 10 \
    --batch_size 8 \
    --model_type temporal \
    --sequence_length 3 \
    --hidden_dim 32 \
    --num_layers 2 \
    --lr 0.001
```

### 2. Standard Training (Balanced)

```bash
python train.py \
    --data_dir BFS-OpenFOAM-data \
    --epochs 50 \
    --batch_size 4 \
    --model_type temporal \
    --sequence_length 3 \
    --hidden_dim 64 \
    --num_layers 3 \
    --lr 0.001
```

### 3. High Performance (Slower, Better Accuracy)

```bash
python train.py \
    --data_dir BFS-OpenFOAM-data \
    --epochs 100 \
    --batch_size 2 \
    --model_type temporal \
    --sequence_length 5 \
    --hidden_dim 128 \
    --num_layers 4 \
    --lr 0.0005
```

### 4. Maximum Performance (Very Slow, Best Accuracy)

```bash
python train.py \
    --data_dir BFS-OpenFOAM-data \
    --epochs 200 \
    --batch_size 2 \
    --model_type temporal \
    --sequence_length 7 \
    --hidden_dim 256 \
    --num_layers 5 \
    --lr 0.0003 \
    --layer_type GAT
```

## Training Process

### What Happens During Training

1. **Data Loading**: Loads OpenFOAM mesh and field data
2. **Graph Construction**: Builds graphs from mesh connectivity
3. **Data Splitting**: Splits into training (80%) and validation (20%)
4. **Training Loop**: 
   - Forward pass through model
   - Compute loss (MSE)
   - Backward pass and optimization
   - Validation after each epoch
5. **Checkpointing**: Saves best model based on validation loss

### Monitoring Training

The training script outputs:
- Training loss per epoch
- Validation loss per epoch
- Best model checkpoint saved automatically
- Training curves plot saved to `checkpoints/training_curves.png`

### Example Output

```
Using device: cpu
Loading OpenFOAM data...
Loading mesh connectivity...
Mesh loaded: 12225 cells, 24170 internal faces
Found 101 time directories
Building graphs...
Built 101 graphs
Train graphs: 80, Val graphs: 21
Model parameters: 92,484

Starting training...

Epoch 1/50
Training: 100%|██████████| 39/39 [00:17<00:00,  2.21it/s]
Validating: 100%|██████████| 9/9 [00:01<00:00,  5.20it/s]
Train Loss: 32133.886475, Val Loss: 23045.761936
Saved best model (val_loss: 23045.761936)

Epoch 2/50
...
```

## Tips for Better Training

### 1. Start Small, Scale Up

Begin with a small model to verify everything works:
```bash
python train.py --epochs 10 --hidden_dim 32 --num_layers 2
```

Then scale up:
```bash
python train.py --epochs 50 --hidden_dim 64 --num_layers 3
```

### 2. Monitor Training Curves

Check `checkpoints/training_curves.png` to see:
- If training loss is decreasing
- If validation loss is decreasing
- If there's overfitting (train loss << val loss)

### 3. Adjust Learning Rate

If loss is not decreasing:
- Try lower learning rate: `--lr 0.0005` or `--lr 0.0001`
- If loss decreases too slowly: try `--lr 0.002`

### 4. Use GPU if Available

The script automatically uses GPU if available. To force CPU:
```python
# Edit train.py and set:
device = torch.device('cpu')
```

### 5. Batch Size Considerations

- Larger batch size = faster training but more memory
- Smaller batch size = slower but can fit larger models
- Adjust based on your system memory

### 6. Sequence Length

For temporal models:
- Shorter sequences (3-5): Faster, less context
- Longer sequences (7-10): Slower, more context, potentially better

## Resuming Training

Currently, the script doesn't support resuming from checkpoint. To continue training:

1. Note the best epoch from previous training
2. Start new training with more epochs
3. The best model will be saved automatically

## Troubleshooting

### Training Loss Not Decreasing

1. **Check learning rate:**
   - Try lower: `--lr 0.0001`
   - Try higher: `--lr 0.002`

2. **Check model capacity:**
   - Increase `--hidden_dim` (64 → 128 → 256)
   - Increase `--num_layers` (3 → 4 → 5)

3. **Check data:**
   ```bash
   python test_parser.py
   ```

### Out of Memory

1. **Reduce batch size:**
   ```bash
   --batch_size 2  # or even 1
   ```

2. **Reduce model size:**
   ```bash
   --hidden_dim 32 --num_layers 2
   ```

3. **Reduce sequence length:**
   ```bash
   --sequence_length 2
   ```

### Training is Too Slow

1. **Increase batch size** (if memory allows)
2. **Reduce sequence length**
3. **Reduce model size**
4. **Use GPU** if available

### Validation Loss Higher Than Training Loss

This is normal - validation loss is typically higher. If the gap is very large:
- Model might be overfitting
- Try adding dropout or reducing model capacity
- Check if training data is representative

## After Training

1. **Check training curves:**
   ```bash
   open checkpoints/training_curves.png
   ```

2. **Generate visualizations:**
   ```bash
   python plot_contours.py --checkpoint checkpoints/best_model.pt --data_dir BFS-OpenFOAM-data
   ```

3. **Evaluate predictions:**
   ```bash
   python plot_fields.py --checkpoint checkpoints/best_model.pt --data_dir BFS-OpenFOAM-data
   ```

## Full Command Reference

```bash
python train.py \
    --data_dir BFS-OpenFOAM-data \
    --epochs 50 \
    --batch_size 4 \
    --lr 0.001 \
    --hidden_dim 64 \
    --num_layers 3 \
    --sequence_length 3 \
    --predict_ahead 1 \
    --model_type temporal \
    --layer_type GCN \
    --train_split 0.8 \
    --save_dir checkpoints
```

## Best Practices

1. **Always check training curves** after training
2. **Start with default parameters** and adjust based on results
3. **Save different model configurations** with different `--save_dir`
4. **Monitor validation loss** - it indicates generalization
5. **Train for enough epochs** - at least 50, preferably 100+

