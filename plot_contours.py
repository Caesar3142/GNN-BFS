"""
Plot contour plots for method-based model.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from pathlib import Path
import argparse

from openfoam_parser import OpenFOAMParser
from graph_constructor_method import GraphConstructorMethod
from gnn_model_method import FlowGNNMethod, TemporalFlowGNNMethod
from normalization import FeatureNormalizer, normalize_graph_features


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint['args']
    
    input_dim = 24  # 4 fields * 6 time steps
    output_dim = 4  # 4 fields: [c, p, u, v]
    
    if args.model_type == 'temporal':
        model = TemporalFlowGNNMethod(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=output_dim,
            num_layers=args.num_layers,
            dropout=0.1
        )
    else:
        model = FlowGNNMethod(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=output_dim,
            num_layers=args.num_layers,
            dropout=0.1
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    normalizer = checkpoint.get('normalizer', None)
    
    return model, args, normalizer


def predict(model, graph, device, normalizer=None):
    """Make prediction."""
    model.eval()
    
    with torch.no_grad():
        if normalizer is not None:
            x_norm = normalizer.transform(graph.x.numpy())
            x = torch.tensor(x_norm, dtype=torch.float32).to(device)
        else:
            x = graph.x.to(device)
        
        edge_index = graph.edge_index.to(device)
        pred = model(x, edge_index)
        
        if normalizer is not None:
            pred = normalizer.inverse_transform(pred.cpu().numpy())
        else:
            pred = pred.cpu().numpy()
    
    return pred


def compute_cell_centers(parser):
    """Compute cell center coordinates."""
    return parser.compute_cell_centers()


def plot_contour_comparison(x, y, gt_values, pred_values, title_base, save_path=None, levels=20):
    """Plot vertically stacked comparison of ground truth and prediction contours."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))
    
    # Create grid
    xi = np.linspace(x.min(), x.max(), 200)
    yi = np.linspace(y.min(), y.max(), 200)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Interpolate to grid
    zi_gt = griddata((x, y), gt_values, (xi_grid, yi_grid), method='linear', fill_value=np.nan)
    zi_pred = griddata((x, y), pred_values, (xi_grid, yi_grid), method='linear', fill_value=np.nan)
    
    # Calculate percentage error
    gt_magnitude = np.abs(gt_values)
    threshold = np.max(gt_magnitude) * 1e-6
    error_percent = np.where(gt_magnitude > threshold,
                            ((pred_values - gt_values) / gt_magnitude) * 100,
                            np.zeros_like(gt_values))
    zi_error = griddata((x, y), error_percent, (xi_grid, yi_grid), method='linear', fill_value=np.nan)
    
    # Domain masking with Delaunay
    try:
        points_2d = np.column_stack([x, y])
        tri = Delaunay(points_2d)
        grid_points = np.column_stack([xi_grid.ravel(), yi_grid.ravel()])
        mask = tri.find_simplex(grid_points) >= 0
        mask = mask.reshape(xi_grid.shape)
        
        zi_gt[~mask] = np.nan
        zi_pred[~mask] = np.nan
        zi_error[~mask] = np.nan
    except:
        pass
    
    # Common scale for OpenFOAM and prediction
    vmin = min(np.nanmin(zi_gt), np.nanmin(zi_pred))
    vmax = max(np.nanmax(zi_gt), np.nanmax(zi_pred))
    level_values = np.linspace(vmin, vmax, levels)
    
    # Plot 1: OpenFOAM - Top
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
    cbar1.set_ticks(np.linspace(vmin, vmax, 6))
    
    # Plot 2: Prediction - Middle
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
    cbar2.set_ticks(np.linspace(vmin, vmax, 6))
    
    # Plot 3: Percentage Error - Bottom
    error_max = max(abs(np.nanmin(zi_error)), abs(np.nanmax(zi_error)))
    error_levels = np.linspace(-error_max, error_max, levels)
    contour3 = axes[2].contourf(xi_grid, yi_grid, zi_error, levels=error_levels,
                               vmin=-error_max, vmax=error_max, cmap='RdBu_r', extend='neither')
    axes[2].contour(xi_grid, yi_grid, zi_error, levels=error_levels, colors='black',
                   alpha=0.3, linewidths=0.5)
    axes[2].set_xlabel('X coordinate', fontsize=12)
    axes[2].set_ylabel('Y coordinate', fontsize=12)
    axes[2].set_title(f'Percentage Error: (Predicted - OpenFOAM) / OpenFOAM × 100%', fontsize=14, fontweight='bold')
    axes[2].set_aspect('equal')
    axes[2].grid(True, alpha=0.3)
    cbar3 = plt.colorbar(contour3, ax=axes[2], label='Error (%)', fraction=0.046, pad=0.04)
    cbar3.set_ticks(np.linspace(-error_max, error_max, 6))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Contour comparison saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot contour plots for method-based model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='BFS-OpenFOAM-data',
                       help='Path to OpenFOAM data directory')
    parser.add_argument('--time_dir', type=str, default=None,
                       help='Time directory to plot (default: validation time step)')
    parser.add_argument('--output_dir', type=str, default='contour_plots_method',
                       help='Directory to save plots')
    parser.add_argument('--levels', type=int, default=30,
                       help='Number of contour levels')
    parser.add_argument('--include_concentration', action='store_true',
                       help='Include concentration field (if available)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    print("Loading model...")
    model, model_args, normalizer = load_model(args.checkpoint, device)
    print(f"Model loaded: {model_args.model_type}")
    if normalizer is not None:
        print("Normalizer found in checkpoint")
    
    # Load data
    print("Loading OpenFOAM data...")
    of_parser = OpenFOAMParser(args.data_dir)
    gc = GraphConstructorMethod(of_parser)
    
    # Get time directories
    data_path = Path(args.data_dir)
    time_dirs = sorted([d.name for d in data_path.iterdir() 
                       if d.is_dir() and d.name.replace('.', '').isdigit()])
    
    # Determine which time step to plot
    if args.time_dir:
        if args.time_dir not in time_dirs:
            raise ValueError(f"Time directory {args.time_dir} not found")
        plot_time = args.time_dir
    else:
        plot_time = time_dirs[-5]  # Use validation time step
    
    print(f"Plotting for time: {plot_time}")
    
    # Compute cell centers
    print("Computing cell center coordinates...")
    cell_centers = compute_cell_centers(of_parser)
    x = cell_centers[:, 0]
    y = cell_centers[:, 1]
    
    # Need 6 previous time steps for input
    time_idx = time_dirs.index(plot_time)
    
    if time_idx < 6:
        raise ValueError(f"Need at least 6 previous time steps. Only {time_idx} available before {plot_time}")
    
    # Build input sequence (6 time steps)
    input_sequence = []
    for i in range(time_idx - 6, time_idx):
        graph = gc.build_graph(
            time_dirs[i],
            adaptive_sampling=model_args.adaptive_sampling if hasattr(model_args, 'adaptive_sampling') else False,
            include_concentration=args.include_concentration
        )
        if normalizer is not None:
            graph_list, _ = normalize_graph_features([graph], normalizer=normalizer, fit=False)
            graph = graph_list[0]
        input_sequence.append(graph)
    
    # Flatten sequence
    flattened_features = torch.cat([g.x for g in input_sequence], dim=1)
    input_graph = type(input_sequence[0])(
        x=flattened_features,
        edge_index=input_sequence[0].edge_index,
        num_nodes=input_sequence[0].num_nodes
    )
    
    # Make prediction
    print("Making prediction...")
    prediction = predict(model, input_graph, device, normalizer)
    
    # Get ground truth (next time step)
    target_time_idx = time_idx + 1
    if target_time_idx < len(time_dirs):
        target_time = time_dirs[target_time_idx]
        target_graph = gc.build_graph(
            target_time,
            adaptive_sampling=model_args.adaptive_sampling if hasattr(model_args, 'adaptive_sampling') else False,
            include_concentration=args.include_concentration
        )
        ground_truth = target_graph.x.numpy()
    else:
        print("Warning: No next time step, using current")
        target_graph = gc.build_graph(
            plot_time,
            adaptive_sampling=model_args.adaptive_sampling if hasattr(model_args, 'adaptive_sampling') else False,
            include_concentration=args.include_concentration
        )
        ground_truth = target_graph.x.numpy()
    
    # Extract fields: [c, p, u, v]
    gt_pressure = ground_truth[:, 1]  # p (index 1)
    pred_pressure = prediction[:, 1]
    
    gt_velocity_u = ground_truth[:, 2]  # u (index 2)
    gt_velocity_v = ground_truth[:, 3]  # v (index 3)
    pred_velocity_u = prediction[:, 2]
    pred_velocity_v = prediction[:, 3]
    
    gt_velocity_mag = np.sqrt(gt_velocity_u**2 + gt_velocity_v**2)
    pred_velocity_mag = np.sqrt(pred_velocity_u**2 + pred_velocity_v**2)
    
    print(f"\nCreating contour plots...")
    
    # Plot pressure contours
    print("Creating pressure contour plots...")
    plot_contour_comparison(
        x, y, gt_pressure, pred_pressure,
        'Pressure',
        save_path=output_dir / f"pressure_contours_{plot_time}.png",
        levels=args.levels
    )
    
    # Plot velocity contours
    print("Creating velocity contour plots...")
    plot_contour_comparison(
        x, y, gt_velocity_mag, pred_velocity_mag,
        'Velocity Magnitude',
        save_path=output_dir / f"velocity_contours_{plot_time}.png",
        levels=args.levels
    )
    
    print(f"\nAll contour plots saved to {output_dir}/")


if __name__ == '__main__':
    main()

