"""
Plot pressure and velocity fields from OpenFOAM data and trained model predictions.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
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


def plot_pressure_comparison(gt_pressure, pred_pressure, save_path=None):
    """Plot pressure comparison between OpenFOAM and model predictions."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Scatter plot: predicted vs ground truth
    axes[0, 0].scatter(gt_pressure, pred_pressure, alpha=0.4, s=2, c='blue', edgecolors='none')
    min_val = min(gt_pressure.min(), pred_pressure.min())
    max_val = max(gt_pressure.max(), pred_pressure.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    axes[0, 0].set_xlabel('OpenFOAM Pressure', fontsize=12)
    axes[0, 0].set_ylabel('Model Predicted Pressure', fontsize=12)
    axes[0, 0].set_title('Pressure: Predicted vs OpenFOAM', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Calculate R²
    ss_res = np.sum((gt_pressure - pred_pressure) ** 2)
    ss_tot = np.sum((gt_pressure - np.mean(gt_pressure)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    mae = np.mean(np.abs(gt_pressure - pred_pressure))
    rmse = np.sqrt(np.mean((gt_pressure - pred_pressure) ** 2))
    
    axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}', 
                    transform=axes[0, 0].transAxes,
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=10)
    
    # 2. Error distribution
    error = pred_pressure - gt_pressure
    axes[0, 1].hist(error, bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[0, 1].set_xlabel('Pressure Error (Predicted - OpenFOAM)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Pressure Error Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(0, color='r', linestyle='--', linewidth=2, label='Zero error')
    axes[0, 1].axvline(np.mean(error), color='g', linestyle='--', linewidth=2, 
                      label=f'Mean: {np.mean(error):.4f}')
    axes[0, 1].legend()
    
    # 3. Absolute error distribution
    abs_error = np.abs(error)
    axes[1, 0].hist(abs_error, bins=50, alpha=0.7, edgecolor='black', color='red')
    axes[1, 0].set_xlabel('Absolute Pressure Error', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Absolute Pressure Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(np.mean(abs_error), color='g', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(abs_error):.4f}')
    axes[1, 0].axvline(np.median(abs_error), color='b', linestyle='--', linewidth=2,
                      label=f'Median: {np.median(abs_error):.4f}')
    axes[1, 0].legend()
    
    # 4. Relative error vs magnitude
    relative_error = np.abs(error) / (np.abs(gt_pressure) + 1e-10) * 100
    axes[1, 1].scatter(gt_pressure, relative_error, alpha=0.3, s=1, c='purple', edgecolors='none')
    axes[1, 1].set_xlabel('OpenFOAM Pressure', fontsize=12)
    axes[1, 1].set_ylabel('Relative Error (%)', fontsize=12)
    axes[1, 1].set_title('Relative Pressure Error vs Magnitude', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(np.mean(relative_error), color='r', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(relative_error):.2f}%')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Pressure comparison saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_velocity_comparison(gt_velocity, pred_velocity, save_path=None):
    """Plot velocity comparison between OpenFOAM and model predictions."""
    # Calculate velocity magnitude
    gt_mag = np.linalg.norm(gt_velocity, axis=1)
    pred_mag = np.linalg.norm(pred_velocity, axis=1)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Velocity magnitude: predicted vs ground truth
    axes[0, 0].scatter(gt_mag, pred_mag, alpha=0.4, s=2, c='blue', edgecolors='none')
    min_val = min(gt_mag.min(), pred_mag.min())
    max_val = max(gt_mag.max(), pred_mag.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    axes[0, 0].set_xlabel('OpenFOAM Velocity Magnitude', fontsize=12)
    axes[0, 0].set_ylabel('Model Predicted Velocity Magnitude', fontsize=12)
    axes[0, 0].set_title('Velocity Magnitude: Predicted vs OpenFOAM', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Calculate R²
    ss_res = np.sum((gt_mag - pred_mag) ** 2)
    ss_tot = np.sum((gt_mag - np.mean(gt_mag)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    mae = np.mean(np.abs(gt_mag - pred_mag))
    rmse = np.sqrt(np.mean((gt_mag - pred_mag) ** 2))
    
    axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}', 
                    transform=axes[0, 0].transAxes,
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=10)
    
    # 2. Velocity magnitude error distribution
    error_mag = pred_mag - gt_mag
    axes[0, 1].hist(error_mag, bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[0, 1].set_xlabel('Velocity Magnitude Error (Predicted - OpenFOAM)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Velocity Magnitude Error Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(0, color='r', linestyle='--', linewidth=2, label='Zero error')
    axes[0, 1].axvline(np.mean(error_mag), color='g', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(error_mag):.4f}')
    axes[0, 1].legend()
    
    # 3. Absolute velocity magnitude error
    abs_error_mag = np.abs(error_mag)
    axes[0, 2].hist(abs_error_mag, bins=50, alpha=0.7, edgecolor='black', color='red')
    axes[0, 2].set_xlabel('Absolute Velocity Magnitude Error', fontsize=12)
    axes[0, 2].set_ylabel('Frequency', fontsize=12)
    axes[0, 2].set_title('Absolute Velocity Magnitude Error', fontsize=14, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axvline(np.mean(abs_error_mag), color='g', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(abs_error_mag):.4f}')
    axes[0, 2].legend()
    
    # 4. Component-wise: Ux
    ux_error = pred_velocity[:, 0] - gt_velocity[:, 0]
    axes[1, 0].scatter(gt_velocity[:, 0], pred_velocity[:, 0], alpha=0.4, s=2, c='green', edgecolors='none')
    min_ux = min(gt_velocity[:, 0].min(), pred_velocity[:, 0].min())
    max_ux = max(gt_velocity[:, 0].max(), pred_velocity[:, 0].max())
    axes[1, 0].plot([min_ux, max_ux], [min_ux, max_ux], 'r--', lw=2)
    axes[1, 0].set_xlabel('OpenFOAM Ux', fontsize=12)
    axes[1, 0].set_ylabel('Model Predicted Ux', fontsize=12)
    axes[1, 0].set_title('Ux Component', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    r2_ux = 1 - (np.sum(ux_error**2) / np.sum((gt_velocity[:, 0] - np.mean(gt_velocity[:, 0]))**2))
    axes[1, 0].text(0.05, 0.95, f'R² = {r2_ux:.4f}', transform=axes[1, 0].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 5. Component-wise: Uy
    uy_error = pred_velocity[:, 1] - gt_velocity[:, 1]
    axes[1, 1].scatter(gt_velocity[:, 1], pred_velocity[:, 1], alpha=0.4, s=2, c='purple', edgecolors='none')
    min_uy = min(gt_velocity[:, 1].min(), pred_velocity[:, 1].min())
    max_uy = max(gt_velocity[:, 1].max(), pred_velocity[:, 1].max())
    axes[1, 1].plot([min_uy, max_uy], [min_uy, max_uy], 'r--', lw=2)
    axes[1, 1].set_xlabel('OpenFOAM Uy', fontsize=12)
    axes[1, 1].set_ylabel('Model Predicted Uy', fontsize=12)
    axes[1, 1].set_title('Uy Component', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    r2_uy = 1 - (np.sum(uy_error**2) / np.sum((gt_velocity[:, 1] - np.mean(gt_velocity[:, 1]))**2))
    axes[1, 1].text(0.05, 0.95, f'R² = {r2_uy:.4f}', transform=axes[1, 1].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 6. Component-wise: Uz
    uz_error = pred_velocity[:, 2] - gt_velocity[:, 2]
    axes[1, 2].scatter(gt_velocity[:, 2], pred_velocity[:, 2], alpha=0.4, s=2, c='brown', edgecolors='none')
    min_uz = min(gt_velocity[:, 2].min(), pred_velocity[:, 2].min())
    max_uz = max(gt_velocity[:, 2].max(), pred_velocity[:, 2].max())
    axes[1, 2].plot([min_uz, max_uz], [min_uz, max_uz], 'r--', lw=2)
    axes[1, 2].set_xlabel('OpenFOAM Uz', fontsize=12)
    axes[1, 2].set_ylabel('Model Predicted Uz', fontsize=12)
    axes[1, 2].set_title('Uz Component', fontsize=14, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    r2_uz = 1 - (np.sum(uz_error**2) / np.sum((gt_velocity[:, 2] - np.mean(gt_velocity[:, 2]))**2))
    axes[1, 2].text(0.05, 0.95, f'R² = {r2_uz:.4f}', transform=axes[1, 2].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Velocity comparison saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_error_summary(gt_velocity, pred_velocity, gt_pressure, pred_pressure, save_path=None):
    """Plot summary of errors for both velocity and pressure."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Velocity magnitude error
    gt_vmag = np.linalg.norm(gt_velocity, axis=1)
    pred_vmag = np.linalg.norm(pred_velocity, axis=1)
    v_error = np.abs(pred_vmag - gt_vmag)
    
    # Pressure error
    p_error = np.abs(pred_pressure - gt_pressure)
    
    # 1. Error comparison: Velocity vs Pressure
    axes[0, 0].hist(v_error, bins=50, alpha=0.6, label='Velocity Magnitude Error', color='blue', edgecolor='black')
    axes[0, 0].hist(p_error, bins=50, alpha=0.6, label='Pressure Error', color='red', edgecolor='black')
    axes[0, 0].set_xlabel('Absolute Error', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Error Distribution Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Relative error comparison
    v_rel_error = v_error / (gt_vmag + 1e-10) * 100
    p_rel_error = p_error / (np.abs(gt_pressure) + 1e-10) * 100
    axes[0, 1].hist(v_rel_error, bins=50, alpha=0.6, label='Velocity Relative Error (%)', color='blue', edgecolor='black')
    axes[0, 1].hist(p_rel_error, bins=50, alpha=0.6, label='Pressure Relative Error (%)', color='red', edgecolor='black')
    axes[0, 1].set_xlabel('Relative Error (%)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Relative Error Distribution Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Error statistics table
    axes[1, 0].axis('off')
    stats_text = f"""
    VELOCITY MAGNITUDE ERROR STATISTICS
    {'='*40}
    Mean Absolute Error:     {np.mean(v_error):.6f}
    Median Absolute Error:   {np.median(v_error):.6f}
    Max Absolute Error:       {np.max(v_error):.6f}
    Mean Relative Error:      {np.mean(v_rel_error):.2f}%
    RMSE:                     {np.sqrt(np.mean((pred_vmag - gt_vmag)**2)):.6f}
    
    PRESSURE ERROR STATISTICS
    {'='*40}
    Mean Absolute Error:     {np.mean(p_error):.6f}
    Median Absolute Error:   {np.median(p_error):.6f}
    Max Absolute Error:       {np.max(p_error):.6f}
    Mean Relative Error:      {np.mean(p_rel_error):.2f}%
    RMSE:                     {np.sqrt(np.mean((pred_pressure - gt_pressure)**2)):.6f}
    """
    axes[1, 0].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                   verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 4. Error vs field magnitude
    axes[1, 1].scatter(gt_vmag, v_error, alpha=0.3, s=1, label='Velocity Error', color='blue')
    axes[1, 1].scatter(np.abs(gt_pressure), p_error, alpha=0.3, s=1, label='Pressure Error', color='red')
    axes[1, 1].set_xlabel('Field Magnitude (Normalized)', fontsize=12)
    axes[1, 1].set_ylabel('Absolute Error', fontsize=12)
    axes[1, 1].set_title('Error vs Field Magnitude', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Error summary saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot pressure and velocity fields from OpenFOAM and model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='BFS-OpenFOAM-data',
                       help='Path to OpenFOAM data directory')
    parser.add_argument('--time_dir', type=str, default=None,
                       help='Time directory to plot (default: last available)')
    parser.add_argument('--output_dir', type=str, default='field_plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
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
    
    # Determine which time step to plot
    if args.time_dir:
        if args.time_dir not in time_dirs:
            raise ValueError(f"Time directory {args.time_dir} not found")
        plot_time = args.time_dir
    else:
        # Use a time step from validation set
        plot_time = time_dirs[-5]
    
    print(f"Plotting for time: {plot_time}")
    
    # Build graph for prediction
    graph = graph_constructor.build_graph(plot_time)
    
    # For temporal models, need input sequence
    if model_type == 'temporal':
        seq_len = getattr(model_args, 'sequence_length', 3)
        time_idx = time_dirs.index(plot_time)
        
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
    
    # Get ground truth
    ground_truth = graph.x.numpy()
    
    # Extract velocity and pressure
    gt_velocity = ground_truth[:, :3]  # First 3 columns are velocity
    pred_velocity = prediction[:, :3]
    gt_pressure = ground_truth[:, 3] if ground_truth.shape[1] > 3 else np.zeros(len(ground_truth))
    pred_pressure = prediction[:, 3] if prediction.shape[1] > 3 else np.zeros(len(prediction))
    
    print(f"\nExtracted fields:")
    print(f"  Velocity shape: {gt_velocity.shape}")
    print(f"  Pressure shape: {gt_pressure.shape}")
    
    # Create plots
    print("\nCreating pressure comparison plot...")
    plot_pressure_comparison(
        gt_pressure, pred_pressure,
        save_path=output_dir / f"pressure_comparison_{plot_time}.png"
    )
    
    print("Creating velocity comparison plot...")
    plot_velocity_comparison(
        gt_velocity, pred_velocity,
        save_path=output_dir / f"velocity_comparison_{plot_time}.png"
    )
    
    print("Creating error summary plot...")
    plot_error_summary(
        gt_velocity, pred_velocity, gt_pressure, pred_pressure,
        save_path=output_dir / f"error_summary_{plot_time}.png"
    )
    
    print(f"\nAll plots saved to {output_dir}/")


if __name__ == '__main__':
    main()

