# Quick Start Guide: Training and Visualization

## Training

### Method-Based Implementation

```bash
# Activate virtual environment
source venv/bin/activate

# Train the model
python train_method.py \
    --data_dir BFS-OpenFOAM-data \
    --epochs 100 \
    --batch_size 4 \
    --model_type temporal \
    --hidden_dim 64 \
    --num_layers 4 \
    --lr 0.001 \
    --adaptive_sampling \
    --normalize \
    --normalize_per_field
```

**Checkpoints saved to:** `checkpoints_method/best_model.pt`

---

## Visualization

### Contour Plots (Pressure & Velocity in X-Y Coordinates)

```bash
python plot_contours.py \
    --checkpoint checkpoints_method/best_model.pt \
    --data_dir BFS-OpenFOAM-data \
    --time_dir 0.096 \
    --output_dir contour_plots
```

**To update visualization:**
1. Retrain the model (if needed)
2. Run `plot_contours.py` with the new checkpoint
3. Plots are saved to `contour_plots/` directory

### Field Comparison Plots

```bash
python plot_fields.py \
    --checkpoint checkpoints_method/best_model.pt \
    --data_dir BFS-OpenFOAM-data \
    --time_dir 0.096 \
    --output_dir field_plots
```

### All Visualizations

```bash
python visualize.py \
    --checkpoint checkpoints_method/best_model.pt \
    --data_dir BFS-OpenFOAM-data \
    --all \
    --time_dir 0.096
```

---

## Complete Workflow

### 1. Train Model
```bash
source venv/bin/activate
python train_method.py --data_dir BFS-OpenFOAM-data --epochs 100 --model_type temporal --normalize --normalize_per_field --adaptive_sampling
```

### 2. Generate Contour Plots
```bash
python plot_contours.py --checkpoint checkpoints_method/best_model.pt --data_dir BFS-OpenFOAM-data --time_dir 0.096
```

### 3. View Results
```bash
# Open the generated plots
open contour_plots/pressure_contours_0.096.png
open contour_plots/velocity_contours_0.096.png
```

### 4. Update Visualization (After Retraining)
```bash
# Just run plot_contours.py again with the new checkpoint
python plot_contours.py --checkpoint checkpoints_method/best_model.pt --data_dir BFS-OpenFOAM-data --time_dir 0.096
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Train | `python train_method.py --data_dir BFS-OpenFOAM-data --epochs 100 --model_type temporal --normalize --normalize_per_field` |
| Contour plots | `python plot_contours.py --checkpoint checkpoints_method/best_model.pt --data_dir BFS-OpenFOAM-data` |
| Field plots | `python plot_fields.py --checkpoint checkpoints_method/best_model.pt --data_dir BFS-OpenFOAM-data` |
| All visualizations | `python visualize.py --checkpoint checkpoints_method/best_model.pt --data_dir BFS-OpenFOAM-data --all` |

---

## Tips

1. **After retraining**: Always regenerate visualizations with the new checkpoint
2. **Different time steps**: Use `--time_dir` to visualize different time steps
3. **Check training curves**: `open checkpoints_method/training_curves.png`
4. **Compare results**: Visualizations show OpenFOAM, prediction, and error side-by-side
5. **Method-based**: Uses fixed 6-time-step input sequence automatically
6. **Checkpoint location**: Method-based models save to `checkpoints_method/` directory
