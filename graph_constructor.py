"""
Graph construction from OpenFOAM mesh data.
"""
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, Optional
from openfoam_parser import OpenFOAMParser


class GraphConstructor:
    """Constructs PyTorch Geometric graphs from OpenFOAM mesh."""
    
    def __init__(self, parser: OpenFOAMParser):
        """
        Initialize graph constructor.
        
        Args:
            parser: OpenFOAMParser instance
        """
        self.parser = parser
        self._mesh_data = None
        self._load_mesh()
    
    def _load_mesh(self):
        """Load and cache mesh connectivity."""
        print("Loading mesh connectivity...")
        self.owner, self.neighbour = self.parser.parse_connectivity()
        self.points = self.parser.parse_points()
        
        # Get number of cells
        self.n_cells = max(self.owner.max(), self.neighbour.max()) + 1
        print(f"Mesh loaded: {self.n_cells} cells, {len(self.neighbour)} internal faces")
    
    def build_edge_index(self) -> torch.Tensor:
        """
        Build edge index from mesh connectivity.
        
        Returns:
            Edge index tensor of shape (2, n_edges) for bidirectional edges
        """
        # Internal faces create edges between owner and neighbour cells
        # Create bidirectional edges
        edges = []
        
        # Forward edges (owner -> neighbour)
        for i in range(len(self.neighbour)):
            owner_cell = self.owner[i]
            neighbour_cell = self.neighbour[i]
            edges.append([owner_cell, neighbour_cell])
            edges.append([neighbour_cell, owner_cell])  # Bidirectional
        
        # Also add edges from boundary faces (owner cells)
        # For boundary faces, we can add self-loops or connect to nearest neighbors
        n_internal = len(self.neighbour)
        n_total_faces = len(self.owner)
        
        # For boundary faces, connect owner to itself (self-loop) or skip
        # Alternatively, connect boundary cells to their nearest internal neighbors
        # For simplicity, we'll just use internal face connectivity
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Remove duplicate edges (if any)
        edge_index = torch.unique(edge_index, dim=1)
        
        return edge_index
    
    def compute_cell_features(self, time_dir: str, 
                             fields: Optional[Dict[str, str]] = None) -> torch.Tensor:
        """
        Compute node features from field data.
        
        Args:
            time_dir: Time directory name
            fields: Dictionary mapping feature names to field names
                   Default: {'velocity': 'U', 'pressure': 'p'}
        
        Returns:
            Feature tensor of shape (n_cells, n_features)
        """
        if fields is None:
            fields = {'velocity': 'U', 'pressure': 'p'}
        
        features = []
        
        # Parse velocity (3D vector -> 3 features)
        try:
            velocity = self.parser.parse_vector_field(time_dir, fields.get('velocity', 'U'))
            # Ensure it's 2D
            if velocity.ndim == 1:
                velocity = velocity.reshape(-1, 1)
            elif velocity.ndim == 2 and velocity.shape[1] != 3:
                # If it's 2D but wrong shape, reshape
                velocity = velocity.reshape(-1, 3)
            features.append(velocity)
        except Exception as e:
            print(f"Warning: Could not load velocity: {e}")
            # Use zeros as fallback
            velocity = np.zeros((self.n_cells, 3))
            features.append(velocity)
        
        # Parse pressure (scalar -> 1 feature)
        try:
            pressure = self.parser.parse_scalar_field(time_dir, fields.get('pressure', 'p'))
            # Ensure it's 2D
            if pressure.ndim == 1:
                pressure = pressure.reshape(-1, 1)
            features.append(pressure)
        except Exception as e:
            print(f"Warning: Could not load pressure: {e}")
            pressure = np.zeros((self.n_cells, 1))
            features.append(pressure)
        
        # Optionally add more fields
        if 'k' in fields:
            try:
                k = self.parser.parse_scalar_field(time_dir, fields['k'])
                # Ensure it's 2D
                if k.ndim == 1:
                    k = k.reshape(-1, 1)
                features.append(k)
            except:
                pass
        
        # Ensure all features are 2D before concatenation
        for i, feat in enumerate(features):
            if feat.ndim == 1:
                features[i] = feat.reshape(-1, 1)
            elif feat.ndim == 0:
                # Scalar value, expand to array
                features[i] = np.full((self.n_cells, 1), feat)
        
        # Concatenate all features
        feature_array = np.concatenate(features, axis=1)
        
        return torch.tensor(feature_array, dtype=torch.float32)
    
    def build_graph(self, time_dir: str, 
                   fields: Optional[Dict[str, str]] = None,
                   target_fields: Optional[Dict[str, str]] = None) -> Data:
        """
        Build a PyTorch Geometric Data object from OpenFOAM data.
        
        Args:
            time_dir: Time directory name
            fields: Dictionary of input field names
            target_fields: Dictionary of target field names (for supervised learning)
        
        Returns:
            PyTorch Geometric Data object
        """
        # Build edge connectivity
        edge_index = self.build_edge_index()
        
        # Build node features
        x = self.compute_cell_features(time_dir, fields)
        
        # Build target features (if provided)
        y = None
        if target_fields:
            y = self.compute_cell_features(time_dir, target_fields)
        
        # Create graph data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            num_nodes=self.n_cells
        )
        
        return data
    
    def build_temporal_graphs(self, time_dirs: List[str],
                             fields: Optional[Dict[str, str]] = None) -> list:
        """
        Build graphs for multiple time steps.
        
        Args:
            time_dirs: List of time directory names
            fields: Dictionary of field names
        
        Returns:
            List of Data objects
        """
        graphs = []
        for time_dir in time_dirs:
            try:
                graph = self.build_graph(time_dir, fields)
                graphs.append(graph)
            except Exception as e:
                print(f"Warning: Could not build graph for {time_dir}: {e}")
                continue
        
        return graphs

