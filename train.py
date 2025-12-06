"""
Training script for GNN flow prediction model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from openfoam_parser import OpenFOAMParser
from graph_constructor import GraphConstructor
from gnn_model import FlowGNN, TemporalFlowGNN

# Import enhanced models if available
try:
    from gnn_model_enhanced import EnhancedFlowGNN, EnhancedTemporalFlowGNN
    from dataset_enhanced import EnhancedFlowDataset, enhanced_collate_fn
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    EnhancedFlowGNN = None
    EnhancedTemporalFlowGNN = None
    EnhancedFlowDataset = None
    enhanced_collate_fn = None


class FlowDataset(Dataset):
    """Dataset for flow field graphs."""
    
    def __init__(self, graphs, sequence_length=1, predict_ahead=1):
        """
        Initialize dataset.
        
        Args:
            graphs: List of graph Data objects
            sequence_length: Number of input time steps
            predict_ahead: Number of steps ahead to predict
        """
        self.graphs = graphs
        self.sequence_length = sequence_length
        self.predict_ahead = predict_ahead
    
    def __len__(self):
        return len(self.graphs) - self.sequence_length - self.predict_ahead + 1
    
    def __getitem__(self, idx):
        # Get input sequence
        input_graphs = self.graphs[idx:idx + self.sequence_length]
        
        # Get target (future state)
        target_idx = idx + self.sequence_length + self.predict_ahead - 1
        target_graph = self.graphs[target_idx]
        
        return input_graphs, target_graph


def collate_fn(batch):
    """Custom collate function for batching graphs."""
    input_sequences, target_graphs = zip(*batch)
    
    # Batch input sequences
    batched_inputs = []
    for t in range(len(input_sequences[0])):
        graphs_at_t = [seq[t] for seq in input_sequences]
        batched = Batch.from_data_list(graphs_at_t)
        batched_inputs.append(batched)
    
    # Batch targets
    batched_targets = Batch.from_data_list(target_graphs)
    
    return batched_inputs, batched_targets


def train_epoch(model, dataloader, optimizer, criterion, device, use_temporal=False, use_flattened=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_data in tqdm(dataloader, desc="Training"):
        # Handle different batch formats
        if use_flattened:
            # Enhanced dataset: (batched_input_graph, batched_target_graph)
            input_graph, targets = batch_data
            input_graph = input_graph.to(device)
            targets = targets.to(device)
            
            # Forward pass with flattened input
            pred = model(input_graph.x, input_graph.edge_index)
            loss = criterion(pred, targets.x)
        else:
            # Standard dataset: (input_sequences, targets)
            input_sequences, targets = batch_data
            targets = targets.to(device)
            
            # Prepare inputs
            if use_temporal:
                # Stack input sequence: [seq_len, num_nodes, features]
                x_seq = torch.stack([g.x for g in input_sequences], dim=0)
                x_seq = x_seq.to(device)
                edge_index = input_sequences[0].edge_index.to(device)
                
                # Forward pass
                outputs = model(x_seq, edge_index)
                # Take last time step output
                pred = outputs[-1]
            else:
                # Use only last input graph
                input_graph = input_sequences[-1].to(device)
                pred = model(input_graph.x, input_graph.edge_index)
            
            # Compute loss
            loss = criterion(pred, targets.x)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device, use_temporal=False, use_flattened=False):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Validating"):
            if use_flattened:
                # Enhanced dataset format
                input_graph, targets = batch_data
                input_graph = input_graph.to(device)
                targets = targets.to(device)
                
                pred = model(input_graph.x, input_graph.edge_index)
                loss = criterion(pred, targets.x)
            else:
                # Standard dataset format
                input_sequences, targets = batch_data
                targets = targets.to(device)
                
                if use_temporal:
                    x_seq = torch.stack([g.x for g in input_sequences], dim=0)
                    x_seq = x_seq.to(device)
                    edge_index = input_sequences[0].edge_index.to(device)
                    outputs = model(x_seq, edge_index)
                    pred = outputs[-1]
                else:
                    input_graph = input_sequences[-1].to(device)
                    pred = model(input_graph.x, input_graph.edge_index)
                
                loss = criterion(pred, targets.x)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train GNN for flow prediction')
    parser.add_argument('--data_dir', type=str, default='BFS-OpenFOAM-data',
                       help='Path to OpenFOAM data directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of GNN layers')
    parser.add_argument('--sequence_length', type=int, default=3,
                       help='Input sequence length for temporal model')
    parser.add_argument('--predict_ahead', type=int, default=1,
                       help='Steps ahead to predict')
    parser.add_argument('--model_type', type=str, default='temporal',
                       choices=['static', 'temporal'],
                       help='Model type: static or temporal')
    parser.add_argument('--layer_type', type=str, default='GCN',
                       choices=['GCN', 'GAT', 'GIN', 'attention'],
                       help='GNN layer type (attention for custom attention mechanism)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Train/validation split ratio')
    parser.add_argument('--adaptive_sampling', action='store_true',
                       help='Use adaptive spatial sampling for graph edges')
    parser.add_argument('--sampling_radius', type=float, default=None,
                       help='Radius for adaptive sampling (default: auto-computed)')
    parser.add_argument('--include_coordinates', action='store_true',
                       help='Embed cell center coordinates in node features')
    parser.add_argument('--use_enhanced_model', action='store_true',
                       help='Use enhanced model with explicit neighbor aggregation')
    parser.add_argument('--use_flattened_input', action='store_true',
                       help='Use flattened sequence input (cylinder project method)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    Path(args.save_dir).mkdir(exist_ok=True)
    
    # Load data
    print("Loading OpenFOAM data...")
    of_parser = OpenFOAMParser(args.data_dir)
    graph_constructor = GraphConstructor(of_parser)
    
    # Get time directories
    time_dirs = of_parser.get_time_directories()
    print(f"Found {len(time_dirs)} time directories")
    
    # Build graphs
    print("Building graphs...")
    graphs = graph_constructor.build_temporal_graphs(
        time_dirs,
        adaptive_sampling=args.adaptive_sampling,
        radius=args.sampling_radius
    )
    print(f"Built {len(graphs)} graphs")
    
    # If using enhanced model, rebuild with coordinate embedding
    if args.use_enhanced_model or args.use_flattened_input:
        print("Rebuilding graphs with coordinate embedding...")
        graphs = []
        for time_dir in time_dirs:
            try:
                graph = graph_constructor.build_graph(
                    time_dir,
                    adaptive_sampling=args.adaptive_sampling,
                    radius=args.sampling_radius,
                    include_coordinates=args.include_coordinates
                )
                graphs.append(graph)
            except Exception as e:
                print(f"Warning: Could not build graph for {time_dir}: {e}")
                continue
        print(f"Rebuilt {len(graphs)} graphs with coordinate embedding")
    
    if len(graphs) == 0:
        raise ValueError("No graphs were built. Check data directory and field names.")
    
    # Get feature dimensions
    if args.use_flattened_input and ENHANCED_AVAILABLE:
        # For flattened input, input_dim = fields * sequence_length
        fields_per_timestep = graphs[0].x.shape[1]  # Fields at one time step
        input_dim = fields_per_timestep * (args.sequence_length if args.model_type == 'temporal' else 6)
        output_dim = fields_per_timestep  # Output is single time step
        print(f"Flattened input: {fields_per_timestep} fields/step × {args.sequence_length} steps = {input_dim} input dim")
    else:
        input_dim = graphs[0].x.shape[1]
        output_dim = input_dim  # Predict same fields
    
    # Split data
    split_idx = int(len(graphs) * args.train_split)
    train_graphs = graphs[:split_idx]
    val_graphs = graphs[split_idx:]
    
    print(f"Train graphs: {len(train_graphs)}, Val graphs: {len(val_graphs)}")
    
    # Create datasets
    if args.use_flattened_input and ENHANCED_AVAILABLE:
        # Use enhanced dataset with flattened sequence input
        train_dataset = EnhancedFlowDataset(
            train_graphs,
            sequence_length=args.sequence_length if args.model_type == 'temporal' else 6,
            predict_ahead=args.predict_ahead
        )
        val_dataset = EnhancedFlowDataset(
            val_graphs,
            sequence_length=args.sequence_length if args.model_type == 'temporal' else 6,
            predict_ahead=args.predict_ahead
        )
        collate_fn_to_use = enhanced_collate_fn
        print("Using enhanced dataset with flattened sequence input")
    else:
        # Use standard dataset
        train_dataset = FlowDataset(
            train_graphs,
            sequence_length=args.sequence_length if args.model_type == 'temporal' else 1,
            predict_ahead=args.predict_ahead
        )
        val_dataset = FlowDataset(
            val_graphs,
            sequence_length=args.sequence_length if args.model_type == 'temporal' else 1,
            predict_ahead=args.predict_ahead
        )
        collate_fn_to_use = collate_fn
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_to_use
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_to_use
    )
    
    # Create model
    if args.use_enhanced_model and ENHANCED_AVAILABLE:
        if args.model_type == 'temporal':
            if args.use_flattened_input:
                # Enhanced model with flattened input
                fields_per_timestep = graphs[0].x.shape[1]
                model = EnhancedTemporalFlowGNN(
                    fields_per_timestep=fields_per_timestep,
                    sequence_length=args.sequence_length if args.model_type == 'temporal' else 6,
                    hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    dropout=0.1,
                    use_explicit_aggregation=True
                )
            else:
                # Enhanced model with sequence input
                model = EnhancedTemporalFlowGNN(
                    fields_per_timestep=input_dim,
                    sequence_length=args.sequence_length,
                    hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    dropout=0.1,
                    use_explicit_aggregation=True
                )
        else:
            # Use attention layers if specified, otherwise use standard layers
            layer_type = 'attention' if args.use_enhanced_model else args.layer_type
            model = EnhancedFlowGNN(
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                output_dim=output_dim,
                num_layers=args.num_layers,
                dropout=0.1,
                layer_type=layer_type,
                use_explicit_aggregation=True
            )
        print("Using enhanced model with explicit neighbor aggregation")
    elif args.model_type == 'temporal':
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
            layer_type=args.layer_type
        )
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            use_temporal=(args.model_type == 'temporal'),
            use_flattened=args.use_flattened_input
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(
            model, val_loader, criterion, device,
            use_temporal=(args.model_type == 'temporal'),
            use_flattened=args.use_flattened_input
        )
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': args
            }, f"{args.save_dir}/best_model.pt")
            print(f"Saved best model (val_loss: {val_loss:.6f})")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{args.save_dir}/training_curves.png")
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")
    print(f"Training curves saved to {args.save_dir}/training_curves.png")


if __name__ == '__main__':
    main()

