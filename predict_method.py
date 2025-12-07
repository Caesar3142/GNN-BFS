"""
Prediction script following exact method specification.
"""
import torch
import numpy as np
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
    
    # Create model
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
    """Make prediction for a single graph."""
    model.eval()
    
    with torch.no_grad():
        # Normalize if normalizer available
        if normalizer is not None:
            x_norm = normalizer.transform(graph.x.numpy())
            x = torch.tensor(x_norm, dtype=torch.float32).to(device)
        else:
            x = graph.x.to(device)
        
        edge_index = graph.edge_index.to(device)
        pred = model(x, edge_index)
        
        # Denormalize if normalizer available
        if normalizer is not None:
            pred = normalizer.inverse_transform(pred.cpu().numpy())
        else:
            pred = pred.cpu().numpy()
    
    return pred


def compute_l2_relative_error(pred, target):
    """Compute L2 relative error: ||pred - target||_2 / ||target||_2"""
    numerator = np.linalg.norm(pred - target, ord=2)
    denominator = np.linalg.norm(target, ord=2)
    
    if denominator < 1e-10:
        return 0.0
    
    return numerator / denominator


def main():
    parser = argparse.ArgumentParser(description='Predict using method-based GNN model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to OpenFOAM data directory')
    parser.add_argument('--time_dir', type=str, default=None,
                       help='Time directory to predict (default: last available)')
    parser.add_argument('--output_dir', type=str, default='predictions_method',
                       help='Directory to save predictions')
    parser.add_argument('--include_concentration', action='store_true',
                       help='Include concentration field (if available)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model, model_args, normalizer = load_model(args.checkpoint, device)
    print(f"Model loaded: {model_args.model_type}")
    if normalizer is not None:
        print("Normalizer found in checkpoint")
    
    # Load data
    print("Loading OpenFOAM data...")
    parser_obj = OpenFOAMParser(args.data_dir)
    gc = GraphConstructorMethod(parser_obj)
    
    # Get time directories
    data_path = Path(args.data_dir)
    time_dirs = sorted([d.name for d in data_path.iterdir() 
                       if d.is_dir() and d.name.replace('.', '').isdigit()])
    
    # Determine which time step to predict
    if args.time_dir:
        if args.time_dir not in time_dirs:
            raise ValueError(f"Time directory {args.time_dir} not found")
        predict_time = args.time_dir
    else:
        predict_time = time_dirs[-1]
    
    print(f"Predicting for time: {predict_time}")
    
    # Need 6 previous time steps for input
    time_idx = time_dirs.index(predict_time)
    
    if time_idx < 6:
        raise ValueError(f"Need at least 6 previous time steps. Only {time_idx} available before {predict_time}")
    
    # Build input sequence (6 time steps)
    input_sequence = []
    for i in range(time_idx - 6, time_idx):
        graph = gc.build_graph(
            time_dirs[i],
            adaptive_sampling=model_args.adaptive_sampling if hasattr(model_args, 'adaptive_sampling') else False,
            include_concentration=args.include_concentration
        )
        input_sequence.append(graph)
    
    # Flatten sequence: [c, p, u, v]_(t-6:t-1)
    flattened_features = torch.cat([g.x for g in input_sequence], dim=1)
    
    # Create input graph
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
        print("Warning: No next time step available, using current time step")
        target_graph = gc.build_graph(
            predict_time,
            adaptive_sampling=model_args.adaptive_sampling if hasattr(model_args, 'adaptive_sampling') else False,
            include_concentration=args.include_concentration
        )
        ground_truth = target_graph.x.numpy()
    
    # Save predictions
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    np.save(output_dir / f"prediction_{predict_time}.npy", prediction)
    np.save(output_dir / f"ground_truth_{predict_time}.npy", ground_truth)
    print(f"Predictions saved to {args.output_dir}")
    
    # Compute metrics
    mse = np.mean((prediction - ground_truth) ** 2)
    mae = np.mean(np.abs(prediction - ground_truth))
    l2_error = compute_l2_relative_error(prediction, ground_truth)
    
    print(f"\nPrediction Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  L2 Relative Error: {l2_error:.6f}")
    
    # Print field-wise errors
    field_names = ['Concentration (c)', 'Pressure (p)', 'x-Velocity (u)', 'y-Velocity (v)']
    for i, name in enumerate(field_names):
        field_mse = np.mean((prediction[:, i] - ground_truth[:, i]) ** 2)
        field_mae = np.mean(np.abs(prediction[:, i] - ground_truth[:, i]))
        print(f"  {name}: MSE={field_mse:.6f}, MAE={field_mae:.6f}")


if __name__ == '__main__':
    main()

