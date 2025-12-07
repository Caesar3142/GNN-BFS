"""
Training script following exact method specification.

Loss function: L = ||y_pred - y_CFD||_2^2 (MSE)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from openfoam_parser import OpenFOAMParser
from graph_constructor_method import GraphConstructorMethod
from gnn_model_method import FlowGNNMethod, TemporalFlowGNNMethod
from dataset_method import FlowDatasetMethod, collate_fn_method
from normalization import FeatureNormalizer, normalize_graph_features


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_data in tqdm(dataloader, desc="Training"):
        input_graph, targets = batch_data
        input_graph = input_graph.to(device)
        targets = targets.to(device)
        
        # Forward pass
        pred = model(input_graph.x, input_graph.edge_index)
        
        # Loss: L = ||y_pred - y_CFD||_2^2 (MSE)
        loss = criterion(pred, targets.x)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Validating"):
            input_graph, targets = batch_data
            input_graph = input_graph.to(device)
            targets = targets.to(device)
            
            pred = model(input_graph.x, input_graph.edge_index)
            loss = criterion(pred, targets.x)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def compute_l2_relative_error(pred, target):
    """
    Compute L2 relative error as per method specification.
    
    L2_error = ||φ_pred − φ_true||_2 / ||φ_true||_2
    """
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    numerator = np.linalg.norm(pred_np - target_np, ord=2)
    denominator = np.linalg.norm(target_np, ord=2)
    
    if denominator < 1e-10:
        return 0.0
    
    return numerator / denominator


def main():
    parser = argparse.ArgumentParser(description='Train GNN following exact method specification')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to OpenFOAM data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_method',
                       help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of GNN layers')
    parser.add_argument('--model_type', type=str, default='static',
                       choices=['static', 'temporal'],
                       help='Model type: static or temporal')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Training set split ratio')
    parser.add_argument('--normalize', action='store_true',
                       help='Normalize features')
    parser.add_argument('--normalize_per_field', action='store_true',
                       help='Normalize each field separately')
    parser.add_argument('--adaptive_sampling', action='store_true',
                       help='Use adaptive spatial sampling for edges')
    parser.add_argument('--include_concentration', action='store_true',
                       help='Include concentration field (if available)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading OpenFOAM data...")
    parser_obj = OpenFOAMParser(args.data_dir)
    gc = GraphConstructorMethod(parser_obj)
    
    # Get time directories
    data_path = Path(args.data_dir)
    time_dirs = sorted([d.name for d in data_path.iterdir() 
                       if d.is_dir() and d.name.replace('.', '').isdigit()])
    print(f"Found {len(time_dirs)} time directories")
    
    # Build graphs with features [c, p, u, v]
    print("Building graphs with features [c, p, u, v]...")
    graphs = gc.build_temporal_graphs(
        time_dirs,
        adaptive_sampling=args.adaptive_sampling,
        include_concentration=args.include_concentration
    )
    
    if len(graphs) == 0:
        raise ValueError("No graphs were built. Check data directory and field names.")
    
    print(f"Built {len(graphs)} graphs")
    print(f"Feature dimension: {graphs[0].x.shape[1]} (should be 4: [c, p, u, v])")
    
    # Normalize features if requested
    normalizer = None
    if args.normalize:
        print("Normalizing features...")
        train_graphs_temp = graphs[:int(len(graphs) * args.train_split)]
        train_graphs, normalizer = normalize_graph_features(
            train_graphs_temp,
            normalizer=FeatureNormalizer(
                method='standardize',
                normalize_per_field=args.normalize_per_field
            ),
            fit=True
        )
        
        # Normalize validation set
        val_graphs_temp = graphs[int(len(graphs) * args.train_split):]
        val_graphs, _ = normalize_graph_features(
            val_graphs_temp, normalizer=normalizer, fit=False
        )
        
        graphs = train_graphs + val_graphs
    
    # Split data
    split_idx = int(len(graphs) * args.train_split)
    train_graphs = graphs[:split_idx]
    val_graphs = graphs[split_idx:]
    
    print(f"Training graphs: {len(train_graphs)}, Validation graphs: {len(val_graphs)}")
    
    # Create datasets (6 time steps as per method)
    train_dataset = FlowDatasetMethod(train_graphs, sequence_length=6, predict_ahead=1)
    val_dataset = FlowDatasetMethod(val_graphs, sequence_length=6, predict_ahead=1)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, collate_fn=collate_fn_method)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, collate_fn=collate_fn_method)
    
    # Create model
    # Input: 4 fields * 6 time steps = 24
    # Output: 4 fields = 4
    input_dim = 24
    output_dim = 4
    
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
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function: MSE as per method specification
    criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"Train Loss (MSE): {train_loss:.6f}, Val Loss (MSE): {val_loss:.6f}")
        
        # Compute L2 relative error on validation set
        model.eval()
        with torch.no_grad():
            sample_batch = next(iter(val_loader))
            input_graph, targets = sample_batch
            input_graph = input_graph.to(device)
            targets = targets.to(device)
            pred = model(input_graph.x, input_graph.edge_index)
            l2_error = compute_l2_relative_error(pred, targets.x)
            print(f"L2 Relative Error: {l2_error:.6f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'l2_error': l2_error,
                'args': args,
                'normalizer': normalizer
            }, checkpoint_path)
            print(f"Saved checkpoint with val_loss={val_loss:.6f}, L2_error={l2_error:.6f}")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss (MSE)')
    plt.plot(val_losses, label='Val Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves (Method Specification)')
    plt.legend()
    plt.grid(True)
    plt.savefig(checkpoint_dir / 'training_curves.png', dpi=150)
    print(f"\nTraining curves saved to {checkpoint_dir / 'training_curves.png'}")


if __name__ == '__main__':
    main()

