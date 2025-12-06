"""
Plot contour plots of pressure and velocity fields in x-y coordinates.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from matplotlib.path import Path
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


def compute_cell_centers(parser, graph_constructor):
    """
    Compute cell center coordinates from OpenFOAM mesh geometry.
    Uses proper cell center computation from faces and points.
    """
    print("Computing cell centers from mesh geometry...")
    cell_centers = parser.compute_cell_centers()
    return cell_centers


def plot_contour_field(x, y, values, title, save_path=None, levels=20, cmap='viridis'):
    """Plot a contour field."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a regular grid for contour plotting
    xi = np.linspace(x.min(), x.max(), 200)
    yi = np.linspace(y.min(), y.max(), 200)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Interpolate values to grid
    zi = griddata((x, y), values, (xi_grid, yi_grid), method='linear', fill_value=np.nan)
    
    # Create contour plot
    contour = ax.contourf(xi_grid, yi_grid, zi, levels=levels, cmap=cmap, extend='both')
    ax.contour(xi_grid, yi_grid, zi, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(title, fontsize=12)
    
    ax.set_xlabel('X coordinate', fontsize=12)
    ax.set_ylabel('Y coordinate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Contour plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_contour_comparison(x, y, gt_values, pred_values, title_base, save_path=None, levels=20):
    """Plot vertically stacked comparison of ground truth and prediction contours."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))
    
    # Create a regular grid for contour plotting
    xi = np.linspace(x.min(), x.max(), 200)
    yi = np.linspace(y.min(), y.max(), 200)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Interpolate ground truth to grid
    zi_gt = griddata((x, y), gt_values, (xi_grid, yi_grid), method='linear', fill_value=np.nan)
    
    # Interpolate prediction to grid
    zi_pred = griddata((x, y), pred_values, (xi_grid, yi_grid), method='linear', fill_value=np.nan)
    
    # Calculate error
    error = pred_values - gt_values
    zi_error = griddata((x, y), error, (xi_grid, yi_grid), method='linear', fill_value=np.nan)
    
    # Create mask to preserve domain boundaries (especially square corners)
    # Use Delaunay triangulation to identify valid interpolation region
    try:
        from scipy.spatial import Delaunay
        points_2d = np.column_stack([x, y])
        tri = Delaunay(points_2d)
        
        # Check which grid points are inside the triangulation
        grid_points = np.column_stack([xi_grid.ravel(), yi_grid.ravel()])
        mask = tri.find_simplex(grid_points) >= 0
        mask = mask.reshape(xi_grid.shape)
        
        # Apply mask to all interpolated fields
        zi_gt[~mask] = np.nan
        zi_pred[~mask] = np.nan
        zi_error[~mask] = np.nan
    except ImportError:
        # Fallback: use convex hull if Delaunay not available
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points_2d)
            hull_path = Path(points_2d[hull.vertices])
            mask = hull_path.contains_points(grid_points).reshape(xi_grid.shape)
            zi_gt[~mask] = np.nan
            zi_pred[~mask] = np.nan
            zi_error[~mask] = np.nan
        except:
            pass
    
    # Determine common color scale for both OpenFOAM and prediction
    # Use the same vmin and vmax for both plots
    vmin = min(np.nanmin(zi_gt), np.nanmin(zi_pred))
    vmax = max(np.nanmax(zi_gt), np.nanmax(zi_pred))
    
    # Create evenly spaced levels based on the common range
    level_values = np.linspace(vmin, vmax, levels)
    
    # Plot 1: OpenFOAM (Ground Truth) - Top
    contour1 = axes[0].contourf(xi_grid, yi_grid, zi_gt, levels=level_values, 
                                vmin=vmin, vmax=vmax, cmap='viridis', extend='neither')
    axes[0].contour(xi_grid, yi_grid, zi_gt, levels=level_values, colors='black', 
                   alpha=0.3, linewidths=0.5)
    axes[0].set_xlabel('X coordinate', fontsize=12)
    axes[0].set_ylabel('Y coordinate', fontsize=12)
    axes[0].set_title(f'OpenFOAM {title_base}', fontsize=14, fontweight='bold')
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    cbar1 = plt.colorbar(contour1, ax=axes[0], label=title_base, fraction=0.046, pad=0.04)
    cbar1.set_ticks(np.linspace(vmin, vmax, 6))  # Set consistent tick marks
    
    # Plot 2: Model Prediction - Middle
    contour2 = axes[1].contourf(xi_grid, yi_grid, zi_pred, levels=level_values,
                               vmin=vmin, vmax=vmax, cmap='viridis', extend='neither')
    axes[1].contour(xi_grid, yi_grid, zi_pred, levels=level_values, colors='black',
                   alpha=0.3, linewidths=0.5)
    axes[1].set_xlabel('X coordinate', fontsize=12)
    axes[1].set_ylabel('Y coordinate', fontsize=12)
    axes[1].set_title(f'Model Prediction {title_base}', fontsize=14, fontweight='bold')
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    cbar2 = plt.colorbar(contour2, ax=axes[1], label=title_base, fraction=0.046, pad=0.04)
    cbar2.set_ticks(np.linspace(vmin, vmax, 6))  # Set same tick marks as OpenFOAM
    
    # Plot 3: Error - Bottom
    error_max = max(abs(np.nanmin(zi_error)), abs(np.nanmax(zi_error)))
    error_levels = np.linspace(-error_max, error_max, levels)
    contour3 = axes[2].contourf(xi_grid, yi_grid, zi_error, levels=error_levels,
                                vmin=-error_max, vmax=error_max, cmap='RdBu_r', extend='neither')
    axes[2].contour(xi_grid, yi_grid, zi_error, levels=error_levels, colors='black',
                   alpha=0.3, linewidths=0.5)
    axes[2].set_xlabel('X coordinate', fontsize=12)
    axes[2].set_ylabel('Y coordinate', fontsize=12)
    axes[2].set_title(f'Error (Predicted - OpenFOAM)', fontsize=14, fontweight='bold')
    axes[2].set_aspect('equal')
    axes[2].grid(True, alpha=0.3)
    
    # Determine error unit based on field type
    if 'Pressure' in title_base or 'pressure' in title_base.lower():
        error_unit = ' (m²/s²)'
    elif 'Velocity' in title_base or 'velocity' in title_base.lower():
        error_unit = ' (m/s)'
    else:
        error_unit = ''
    
    cbar3 = plt.colorbar(contour3, ax=axes[2], label=f'Error{error_unit}', fraction=0.046, pad=0.04)
    cbar3.set_ticks(np.linspace(-error_max, error_max, 6))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Contour comparison saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_velocity_contours(x, y, gt_velocity, pred_velocity, save_path=None, levels=20):
    """Plot velocity magnitude contours."""
    # Calculate velocity magnitude
    gt_mag = np.linalg.norm(gt_velocity, axis=1)
    pred_mag = np.linalg.norm(pred_velocity, axis=1)
    
    plot_contour_comparison(x, y, gt_mag, pred_mag, 'Velocity Magnitude',
                           save_path=save_path, levels=levels)


def plot_pressure_contours(x, y, gt_pressure, pred_pressure, save_path=None, levels=20):
    """Plot pressure contours."""
    plot_contour_comparison(x, y, gt_pressure, pred_pressure, 'Pressure',
                           save_path=save_path, levels=levels)


def main():
    parser = argparse.ArgumentParser(description='Plot contour plots of pressure and velocity fields')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='BFS-OpenFOAM-data',
                       help='Path to OpenFOAM data directory')
    parser.add_argument('--time_dir', type=str, default=None,
                       help='Time directory to plot (default: last available)')
    parser.add_argument('--output_dir', type=str, default='contour_plots',
                       help='Directory to save plots')
    parser.add_argument('--levels', type=int, default=30,
                       help='Number of contour levels')
    
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
    
    # Compute cell centers (x, y coordinates)
    print("Computing cell center coordinates...")
    cell_centers = compute_cell_centers(of_parser, graph_constructor)
    x = cell_centers[:, 0]
    y = cell_centers[:, 1]
    
    print(f"Cell centers: x range [{x.min():.4f}, {x.max():.4f}], "
          f"y range [{y.min():.4f}, {y.max():.4f}]")
    
    # The model predicts the NEXT time step, so we need to:
    # 1. Use input sequence up to plot_time
    # 2. Predict the next time step (plot_time + 1)
    # 3. Compare with ground truth at the next time step
    
    time_idx = time_dirs.index(plot_time)
    predict_ahead = getattr(model_args, 'predict_ahead', 1)
    
    # Check if we have a next time step
    if time_idx + predict_ahead >= len(time_dirs):
        print(f"Warning: Cannot predict {predict_ahead} step(s) ahead from {plot_time}")
        print(f"Using previous time step for comparison")
        target_time_idx = time_idx - 1
        target_time = time_dirs[target_time_idx]
    else:
        target_time_idx = time_idx + predict_ahead
        target_time = time_dirs[target_time_idx]
    
    print(f"Input time: {plot_time}, Predicting time: {target_time}")
    
    # Build input sequence ending at plot_time
    if model_type == 'temporal':
        seq_len = getattr(model_args, 'sequence_length', 3)
        
        if time_idx < seq_len:
            input_sequence = [graph_constructor.build_graph(time_dirs[i]) 
                            for i in range(max(0, time_idx - seq_len + 1), time_idx + 1)]
        else:
            input_sequence = [graph_constructor.build_graph(time_dirs[i]) 
                            for i in range(time_idx - seq_len + 1, time_idx + 1)]
        
        # Use the last graph in sequence for edge_index
        graph_for_edges = input_sequence[-1]
        prediction = predict(model, graph_for_edges, device, use_temporal=True, 
                           input_sequence=input_sequence)
    else:
        # For static model, use current time step to predict next
        graph = graph_constructor.build_graph(plot_time)
        prediction = predict(model, graph, device, use_temporal=False)
    
    # Get ground truth at the TARGET time (next time step)
    target_graph = graph_constructor.build_graph(target_time)
    ground_truth = target_graph.x.numpy()
    
    # Extract velocity and pressure
    gt_velocity = ground_truth[:, :3]
    pred_velocity = prediction[:, :3]
    gt_pressure = ground_truth[:, 3] if ground_truth.shape[1] > 3 else np.zeros(len(ground_truth))
    pred_pressure = prediction[:, 3] if prediction.shape[1] > 3 else np.zeros(len(prediction))
    
    print(f"\nCreating contour plots...")
    
    # Plot pressure contours
    print("Creating pressure contour plots...")
    plot_pressure_contours(
        x, y, gt_pressure, pred_pressure,
        save_path=output_dir / f"pressure_contours_{plot_time}.png",
        levels=args.levels
    )
    
    # Plot velocity contours
    print("Creating velocity contour plots...")
    plot_velocity_contours(
        x, y, gt_velocity, pred_velocity,
        save_path=output_dir / f"velocity_contours_{plot_time}.png",
        levels=args.levels
    )
    
    print(f"\nAll contour plots saved to {output_dir}/")


if __name__ == '__main__':
    main()

