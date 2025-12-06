"""
Inference script for making predictions with trained GNN model.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from openfoam_parser import OpenFOAMParser
from graph_constructor import GraphConstructor
from gnn_model import FlowGNN, TemporalFlowGNN


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    
    # Determine input/output dimensions from checkpoint or args
    # For now, we'll need to infer from data
    input_dim = args.get('input_dim', 4)  # Default: velocity (3) + pressure (1)
    output_dim = args.get('output_dim', input_dim)
    
    # Create model
    if args.model_type == 'temporal':
        model = TemporalFlowGNN(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=output_dim,
            num_layers=args.num_layers,
            dropout=0.1
        )
    else:
        model = FlowGNN(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=output_dim,
            num_layers=args.num_layers,
            dropout=0.1,
            layer_type=args.get('layer_type', 'GCN')
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
            # Stack input sequence
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


def visualize_prediction(ground_truth, prediction, save_path=None):
    """Visualize prediction vs ground truth."""
    # Extract velocity magnitude
    if ground_truth.shape[1] >= 3:
        gt_velocity_mag = np.linalg.norm(ground_truth[:, :3], axis=1)
        pred_velocity_mag = np.linalg.norm(prediction[:, :3], axis=1)
    else:
        gt_velocity_mag = ground_truth[:, 0]
        pred_velocity_mag = prediction[:, 0]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Velocity magnitude comparison
    axes[0].scatter(gt_velocity_mag, pred_velocity_mag, alpha=0.5, s=1)
    axes[0].plot([gt_velocity_mag.min(), gt_velocity_mag.max()],
                 [gt_velocity_mag.min(), gt_velocity_mag.max()], 'r--', lw=2)
    axes[0].set_xlabel('Ground Truth Velocity Magnitude')
    axes[0].set_ylabel('Predicted Velocity Magnitude')
    axes[0].set_title('Velocity Magnitude Prediction')
    axes[0].grid(True)
    
    # Error distribution
    error = np.abs(gt_velocity_mag - pred_velocity_mag)
    axes[1].hist(error, bins=50, alpha=0.7)
    axes[1].set_xlabel('Absolute Error')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Error Distribution')
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Make predictions with trained GNN model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='BFS-OpenFOAM-data',
                       help='Path to OpenFOAM data directory')
    parser.add_argument('--time_dir', type=str, default=None,
                       help='Time directory to predict (default: last available)')
    parser.add_argument('--output_dir', type=str, default='predictions',
                       help='Directory to save predictions')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization plots')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model, model_args = load_model(args.checkpoint, device)
    print(f"Model loaded: {model_args.model_type}")
    
    # Load data
    print("Loading OpenFOAM data...")
    of_parser = OpenFOAMParser(args.data_dir)
    graph_constructor = GraphConstructor(of_parser)
    
    # Get time directories
    time_dirs = of_parser.get_time_directories()
    
    # Determine which time step to predict
    if args.time_dir:
        if args.time_dir not in time_dirs:
            raise ValueError(f"Time directory {args.time_dir} not found")
        predict_time = args.time_dir
    else:
        # Use last time step
        predict_time = time_dirs[-1]
    
    print(f"Predicting for time: {predict_time}")
    
    # Build graph for prediction
    graph = graph_constructor.build_graph(predict_time)
    
    # For temporal models, need input sequence
    if model_args.model_type == 'temporal':
        seq_len = model_args.sequence_length
        time_idx = time_dirs.index(predict_time)
        
        if time_idx < seq_len:
            print(f"Warning: Not enough history for temporal model. Using available data.")
            input_sequence = [graph_constructor.build_graph(time_dirs[i]) 
                            for i in range(max(0, time_idx - seq_len + 1), time_idx + 1)]
        else:
            input_sequence = [graph_constructor.build_graph(time_dirs[i]) 
                            for i in range(time_idx - seq_len + 1, time_idx + 1)]
        
        # Make prediction
        prediction = predict(model, graph, device, use_temporal=True, 
                           input_sequence=input_sequence)
    else:
        # Make prediction
        prediction = predict(model, graph, device, use_temporal=False)
    
    # Get ground truth (next time step if available)
    ground_truth = graph.x.numpy()
    
    # Save predictions
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    np.save(output_dir / f"prediction_{predict_time}.npy", prediction)
    np.save(output_dir / f"ground_truth_{predict_time}.npy", ground_truth)
    print(f"Predictions saved to {args.output_dir}")
    
    # Compute metrics
    mse = np.mean((prediction - ground_truth) ** 2)
    mae = np.mean(np.abs(prediction - ground_truth))
    
    print(f"\nPrediction Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    
    # Visualize
    if args.visualize:
        visualize_prediction(
            ground_truth, prediction,
            save_path=output_dir / f"visualization_{predict_time}.png"
        )


if __name__ == '__main__':
    main()

