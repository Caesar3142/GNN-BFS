"""
Enhanced visualization script for GNN flow predictions.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse

from openfoam_parser import OpenFOAMParser
from graph_constructor import GraphConstructor
from gnn_model import FlowGNN, TemporalFlowGNN


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint['args']
    
    input_dim = getattr(args, 'input_dim', 4)
    output_dim = getattr(args, 'output_dim', input_dim)
    
    model_type = getattr(args, 'model_type', 'static')
    if model_type == 'temporal':
        model = TemporalFlowGNN(
            input_dim=input_dim,
            hidden_dim=getattr(args, 'hidden_dim', 64),
            output_dim=output_dim,
            num_layers=getattr(args, 'num_layers', 3),
            dropout=0.1
        )
    else:
        model = FlowGNN(
            input_dim=input_dim,
            hidden_dim=getattr(args, 'hidden_dim', 64),
            output_dim=output_dim,
            num_layers=getattr(args, 'num_layers', 3),
            dropout=0.1,
            layer_type=getattr(args, 'layer_type', 'GCN')
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, args


def predict(model, graph, device, use_temporal=False, input_sequence=None):
    """Make prediction for a single graph."""
    model.eval()
    
    with torch.no_grad():
        if use_temporal and input_sequence is not None:
            x_seq = torch.stack([g.x for g in input_sequence], dim=0)
            x_seq = x_seq.to(device)
            edge_index = input_sequence[0].edge_index.to(device)
            outputs = model(x_seq, edge_index)
            pred = outputs[-1]
        else:
            x = graph.x.to(device)
            edge_index = graph.edge_index.to(device)
            pred = model(x, edge_index)
    
    return pred.cpu().numpy()


def visualize_field_comparison(ground_truth, prediction, save_path=None, field_name="Velocity"):
    """Visualize comparison between ground truth and prediction."""
    # Extract velocity magnitude if it's a vector field
    if ground_truth.shape[1] >= 3:
        gt_velocity = ground_truth[:, :3]
        pred_velocity = prediction[:, :3]
        gt_mag = np.linalg.norm(gt_velocity, axis=1)
        pred_mag = np.linalg.norm(pred_velocity, axis=1)
        field_label = "Velocity Magnitude"
    else:
        gt_mag = ground_truth[:, 0]
        pred_mag = prediction[:, 0]
        field_label = field_name
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Scatter plot: predicted vs ground truth
    axes[0, 0].scatter(gt_mag, pred_mag, alpha=0.3, s=1)
    min_val = min(gt_mag.min(), pred_mag.min())
    max_val = max(gt_mag.max(), pred_mag.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    axes[0, 0].set_xlabel(f'Ground Truth {field_label}')
    axes[0, 0].set_ylabel(f'Predicted {field_label}')
    axes[0, 0].set_title(f'{field_label} Prediction')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Calculate R²
    ss_res = np.sum((gt_mag - pred_mag) ** 2)
    ss_tot = np.sum((gt_mag - np.mean(gt_mag)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0, 0].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Error distribution
    error = np.abs(gt_mag - pred_mag)
    relative_error = error / (np.abs(gt_mag) + 1e-10) * 100  # Percentage
    
    axes[0, 1].hist(error, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Absolute Error')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Absolute Error Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(np.mean(error), color='r', linestyle='--', label=f'Mean: {np.mean(error):.4f}')
    axes[0, 1].legend()
    
    # 3. Relative error distribution
    axes[1, 0].hist(relative_error, bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[1, 0].set_xlabel('Relative Error (%)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Relative Error Distribution (%)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(np.mean(relative_error), color='r', linestyle='--', 
                      label=f'Mean: {np.mean(relative_error):.2f}%')
    axes[1, 0].legend()
    
    # 4. Error vs magnitude
    axes[1, 1].scatter(gt_mag, error, alpha=0.3, s=1)
    axes[1, 1].set_xlabel(f'Ground Truth {field_label}')
    axes[1, 1].set_ylabel('Absolute Error')
    axes[1, 1].set_title('Error vs Magnitude')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_vector_field(velocity_field, points, save_path=None, title="Velocity Field"):
    """Visualize 2D velocity field (projection)."""
    if points.shape[1] != 3:
        print("Warning: Points must be 3D for visualization")
        return
    
    # Project to 2D (use x-y plane, average z)
    x = points[:, 0]
    y = points[:, 1]
    
    # Extract velocity components
    if velocity_field.shape[1] >= 2:
        u = velocity_field[:, 0]
        v = velocity_field[:, 1]
        mag = np.linalg.norm(velocity_field[:, :2], axis=1)
    else:
        print("Warning: Not enough velocity components")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Velocity magnitude contour
    scatter1 = axes[0].scatter(x, y, c=mag, cmap='viridis', s=1, alpha=0.6)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title(f'{title} - Magnitude')
    axes[0].set_aspect('equal')
    plt.colorbar(scatter1, ax=axes[0], label='Velocity Magnitude')
    
    # 2. Velocity vectors (subsampled for clarity)
    subsample = max(1, len(x) // 1000)  # Show ~1000 vectors
    x_sub = x[::subsample]
    y_sub = y[::subsample]
    u_sub = u[::subsample]
    v_sub = v[::subsample]
    mag_sub = mag[::subsample]
    
    axes[1].quiver(x_sub, y_sub, u_sub, v_sub, mag_sub, cmap='viridis', 
                   scale=50, width=0.002, alpha=0.6)
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_title(f'{title} - Vector Field')
    axes[1].set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Vector field visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_component_wise(ground_truth, prediction, save_path=None):
    """Visualize each component separately."""
    if ground_truth.shape[1] < 3:
        print("Not enough components for component-wise visualization")
        return
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    components = ['Ux', 'Uy', 'Uz']
    
    for i in range(3):
        gt_comp = ground_truth[:, i]
        pred_comp = prediction[:, i]
        
        # Scatter plot
        axes[i, 0].scatter(gt_comp, pred_comp, alpha=0.3, s=1)
        min_val = min(gt_comp.min(), pred_comp.min())
        max_val = max(gt_comp.max(), pred_comp.max())
        axes[i, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[i, 0].set_xlabel(f'Ground Truth {components[i]}')
        axes[i, 0].set_ylabel(f'Predicted {components[i]}')
        axes[i, 0].set_title(f'{components[i]} Prediction')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Calculate R²
        ss_res = np.sum((gt_comp - pred_comp) ** 2)
        ss_tot = np.sum((gt_comp - np.mean(gt_comp)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        axes[i, 0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[i, 0].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Error distribution
        error = np.abs(gt_comp - pred_comp)
        axes[i, 1].hist(error, bins=50, alpha=0.7, edgecolor='black')
        axes[i, 1].set_xlabel('Absolute Error')
        axes[i, 1].set_ylabel('Frequency')
        axes[i, 1].set_title(f'{components[i]} Error Distribution')
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].axvline(np.mean(error), color='r', linestyle='--', 
                          label=f'Mean: {np.mean(error):.4f}')
        axes[i, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Component-wise visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize GNN flow predictions')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='BFS-OpenFOAM-data',
                       help='Path to OpenFOAM data directory')
    parser.add_argument('--time_dir', type=str, default=None,
                       help='Time directory to visualize (default: last available)')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--visualize_field', action='store_true',
                       help='Create field comparison plots')
    parser.add_argument('--visualize_vectors', action='store_true',
                       help='Create vector field plots')
    parser.add_argument('--visualize_components', action='store_true',
                       help='Create component-wise plots')
    parser.add_argument('--all', action='store_true',
                       help='Create all visualizations')
    
    args = parser.parse_args()
    
    if args.all:
        args.visualize_field = True
        args.visualize_vectors = True
        args.visualize_components = True
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    print("Loading model...")
    model, model_args = load_model(args.checkpoint, device)
    model_type = getattr(model_args, 'model_type', 'static')
    print(f"Model loaded: {model_type}")
    
    # Load data
    print("Loading OpenFOAM data...")
    of_parser = OpenFOAMParser(args.data_dir)
    graph_constructor = GraphConstructor(of_parser)
    
    # Get time directories
    time_dirs = of_parser.get_time_directories()
    
    # Determine which time step to visualize
    if args.time_dir:
        if args.time_dir not in time_dirs:
            raise ValueError(f"Time directory {args.time_dir} not found")
        predict_time = args.time_dir
    else:
        # Use a time step from validation set (not the last one)
        predict_time = time_dirs[-5]  # Use a time step near the end
    
    print(f"Visualizing for time: {predict_time}")
    
    # Build graph for prediction
    graph = graph_constructor.build_graph(predict_time)
    
    # For temporal models, need input sequence
    model_type = getattr(model_args, 'model_type', 'static')
    if model_type == 'temporal':
        seq_len = getattr(model_args, 'sequence_length', 3)
        time_idx = time_dirs.index(predict_time)
        
        if time_idx < seq_len:
            input_sequence = [graph_constructor.build_graph(time_dirs[i]) 
                            for i in range(max(0, time_idx - seq_len + 1), time_idx + 1)]
        else:
            input_sequence = [graph_constructor.build_graph(time_dirs[i]) 
                            for i in range(time_idx - seq_len + 1, time_idx + 1)]
        
        prediction = predict(model, graph, device, use_temporal=True, 
                           input_sequence=input_sequence)
    else:
        prediction = predict(model, graph, device, use_temporal=False)
    
    # Get ground truth (next time step if available)
    ground_truth = graph.x.numpy()
    
    # Compute metrics
    mse = np.mean((prediction - ground_truth) ** 2)
    mae = np.mean(np.abs(prediction - ground_truth))
    rmse = np.sqrt(mse)
    
    print(f"\nPrediction Metrics:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    
    # Create visualizations
    if args.visualize_field:
        print("\nCreating field comparison plots...")
        visualize_field_comparison(
            ground_truth, prediction,
            save_path=output_dir / f"field_comparison_{predict_time}.png"
        )
    
    if args.visualize_components:
        print("Creating component-wise plots...")
        visualize_component_wise(
            ground_truth, prediction,
            save_path=output_dir / f"components_{predict_time}.png"
        )
    
    if args.visualize_vectors:
        print("Creating vector field plots...")
        # Get mesh points
        points = of_parser.parse_points()
        # Note: points are for all mesh points, but we need cell centers
        # For now, we'll use a simplified visualization
        if ground_truth.shape[1] >= 3:
            visualize_vector_field(
                prediction[:, :3], points[:len(prediction)],
                save_path=output_dir / f"vector_field_{predict_time}.png",
                title="Predicted Velocity Field"
            )
            visualize_vector_field(
                ground_truth[:, :3], points[:len(ground_truth)],
                save_path=output_dir / f"vector_field_gt_{predict_time}.png",
                title="Ground Truth Velocity Field"
            )
    
    print(f"\nVisualizations saved to {output_dir}/")


if __name__ == '__main__':
    main()

