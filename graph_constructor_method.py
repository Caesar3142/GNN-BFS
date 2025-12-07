"""
Graph construction from OpenFOAM mesh data following the method specification.
Features: [c, p, u, v] where c=concentration, p=pressure, u=x-velocity, v=y-velocity
Input: 6 previous time steps flattened per node
"""
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, Optional, List
from openfoam_parser import OpenFOAMParser


class GraphConstructorMethod:
    """Constructs graphs following the exact method specification."""
    
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
    
    def build_edge_index(self, adaptive_sampling=False, radius=None, 
                        cell_centers=None, gradient_info=None) -> torch.Tensor:
        """
        Build edge index using adaptive neighbor sampling.
        
        Neighbor set for node i: ne(i) = { j | (x_i − x_j)² + (y_i − y_j)² ≤ r }
        
        Args:
            adaptive_sampling: If True, use adaptive spatial sampling based on radius
            radius: Sampling radius (if None, computed adaptively)
            cell_centers: Cell center coordinates for adaptive sampling [n_cells, 3]
            gradient_info: Gradient information for adaptive sampling (optional)
        
        Returns:
            Edge index tensor of shape (2, n_edges) for bidirectional edges
        """
        if adaptive_sampling and cell_centers is not None:
            return self._build_adaptive_edge_index(cell_centers, radius, gradient_info)
        else:
            return self._build_mesh_based_edge_index()
    
    def _build_mesh_based_edge_index(self) -> torch.Tensor:
        """Build edges from mesh connectivity (original method)."""
        edges = []
        
        # Internal faces create edges between owner and neighbour cells
        for i in range(len(self.neighbour)):
            owner_cell = self.owner[i]
            neighbour_cell = self.neighbour[i]
            edges.append([owner_cell, neighbour_cell])
            edges.append([neighbour_cell, owner_cell])  # Bidirectional
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index = torch.unique(edge_index, dim=1)
        
        return edge_index
    
    def _build_adaptive_edge_index(self, cell_centers, radius=None, gradient_info=None) -> torch.Tensor:
        """
        Build edges using adaptive spatial sampling.
        
        Formula: ne(i) = { j | (x_i − x_j)² + (y_i − y_j)² ≤ r }
        """
        n_cells = len(cell_centers)
        edges = []
        
        # Compute adaptive radius if not provided
        if radius is None:
            mesh_edges = self._build_mesh_based_edge_index()
            mesh_distances = []
            
            sample_size = min(100, n_cells)
            sample_indices = np.linspace(0, n_cells - 1, sample_size, dtype=int)
            
            for i in sample_indices:
                neighbors = mesh_edges[1, mesh_edges[0] == i]
                if len(neighbors) > 0:
                    neighbor_indices = neighbors.numpy()
                    dists = np.linalg.norm(
                        cell_centers[neighbor_indices] - cell_centers[i], axis=1
                    )
                    mesh_distances.extend(dists.tolist())
            
            if mesh_distances:
                radius = np.median(mesh_distances) * 1.5
            else:
                domain_size = np.linalg.norm(
                    cell_centers.max(axis=0) - cell_centers.min(axis=0)
                )
                radius = domain_size * 0.1
        
        print(f"Using adaptive sampling with radius: {radius:.6f}")
        
        # Use 2D distance (x, y) for 2D flow
        cell_centers_2d = cell_centers[:, :2]
        
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(cell_centers_2d)
            pairs = tree.query_pairs(radius, output_type='ndarray')
            
            for i, j in pairs:
                edges.append([int(i), int(j)])
                edges.append([int(j), int(i)])
        except Exception as e:
            print(f"Warning: cKDTree failed, falling back to brute force: {e}")
            from scipy.spatial.distance import cdist
            for i in range(n_cells):
                distances = cdist(cell_centers_2d[i:i+1], cell_centers_2d)[0]
                neighbors_in_radius = np.where(distances < radius)[0]
                for j in neighbors_in_radius:
                    if i != j:
                        edges.append([i, j])
                        edges.append([j, i])
        
        # Combine with mesh-based edges
        mesh_edges = self._build_mesh_based_edge_index().t().tolist()
        edges.extend(mesh_edges)
        
        # Remove duplicates
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index = torch.unique(edge_index, dim=1)
        
        print(f"Adaptive sampling: {edge_index.shape[1]} edges")
        
        return edge_index
    
    def compute_cell_features(self, time_dir: str, 
                             fields: Optional[Dict[str, str]] = None,
                             include_concentration: bool = False) -> torch.Tensor:
        """
        Compute node features following method specification: [c, p, u, v]
        
        Args:
            time_dir: Time directory name
            fields: Dictionary mapping feature names to field names
                   Default: {'concentration': 'C', 'velocity': 'U', 'pressure': 'p'}
            include_concentration: If True, try to load concentration field
        
        Returns:
            Feature tensor of shape (n_cells, 4) with [c, p, u, v]
        """
        if fields is None:
            fields = {'velocity': 'U', 'pressure': 'p'}
            if include_concentration:
                fields['concentration'] = 'C'
        
        features = []
        
        # 1. Concentration (c) - scalar field
        if include_concentration and 'concentration' in fields:
            try:
                concentration = self.parser.parse_scalar_field(time_dir, fields['concentration'])
                if concentration.ndim == 1:
                    concentration = concentration.reshape(-1, 1)
                features.append(concentration)
            except Exception as e:
                print(f"Warning: Could not load concentration, using zeros: {e}")
                concentration = np.zeros((self.n_cells, 1))
                features.append(concentration)
        else:
            # If concentration not available, use zeros
            concentration = np.zeros((self.n_cells, 1))
            features.append(concentration)
        
        # 2. Pressure (p) - scalar field
        try:
            pressure = self.parser.parse_scalar_field(time_dir, fields.get('pressure', 'p'))
            if pressure.ndim == 1:
                pressure = pressure.reshape(-1, 1)
            features.append(pressure)
        except Exception as e:
            print(f"Warning: Could not load pressure: {e}")
            pressure = np.zeros((self.n_cells, 1))
            features.append(pressure)
        
        # 3. x-velocity (u) and y-velocity (v) - from 3D velocity field
        try:
            velocity = self.parser.parse_vector_field(time_dir, fields.get('velocity', 'U'))
            if velocity.ndim == 1:
                velocity = velocity.reshape(-1, 3)
            elif velocity.ndim == 2 and velocity.shape[1] != 3:
                velocity = velocity.reshape(-1, 3)
            
            # Extract u (x-direction) and v (y-direction) only
            u = velocity[:, 0:1]  # x-velocity
            v = velocity[:, 1:2]  # y-velocity
            features.append(u)
            features.append(v)
        except Exception as e:
            print(f"Warning: Could not load velocity: {e}")
            u = np.zeros((self.n_cells, 1))
            v = np.zeros((self.n_cells, 1))
            features.append(u)
            features.append(v)
        
        # Ensure all features are 2D
        for i, feat in enumerate(features):
            if feat.ndim == 1:
                features[i] = feat.reshape(-1, 1)
            elif feat.ndim == 0:
                features[i] = np.full((self.n_cells, 1), feat)
        
        # Concatenate: [c, p, u, v]
        feature_array = np.concatenate(features, axis=1)
        
        # Should be shape (n_cells, 4)
        assert feature_array.shape[1] == 4, f"Expected 4 features [c, p, u, v], got {feature_array.shape[1]}"
        
        return torch.tensor(feature_array, dtype=torch.float32)
    
    def build_graph(self, time_dir: str, 
                   fields: Optional[Dict[str, str]] = None,
                   adaptive_sampling: bool = False,
                   radius: Optional[float] = None,
                   include_concentration: bool = False) -> Data:
        """
        Build a PyTorch Geometric Data object following method specification.
        
        Args:
            time_dir: Time directory name
            fields: Dictionary of field names
            adaptive_sampling: If True, use adaptive spatial sampling for edges
            radius: Sampling radius for adaptive method
            include_concentration: If True, try to load concentration field
        
        Returns:
            PyTorch Geometric Data object with features [c, p, u, v]
        """
        # Get cell centers for adaptive sampling if needed
        cell_centers = None
        if adaptive_sampling:
            cell_centers = self.parser.compute_cell_centers()
        
        # Build edge connectivity
        edge_index = self.build_edge_index(
            adaptive_sampling=adaptive_sampling,
            radius=radius,
            cell_centers=cell_centers
        )
        
        # Build node features: [c, p, u, v]
        x = self.compute_cell_features(time_dir, fields, include_concentration=include_concentration)
        
        # Create graph data object
        data = Data(
            x=x,
            edge_index=edge_index,
            num_nodes=self.n_cells
        )
        
        return data
    
    def build_temporal_graphs(self, time_dirs: List[str],
                             fields: Optional[Dict[str, str]] = None,
                             adaptive_sampling: bool = False,
                             radius: Optional[float] = None,
                             include_concentration: bool = False) -> list:
        """
        Build graphs for multiple time steps.
        
        Args:
            time_dirs: List of time directory names
            fields: Dictionary of field names
            adaptive_sampling: If True, use adaptive spatial sampling for edges
            radius: Sampling radius for adaptive method
            include_concentration: If True, try to load concentration field
        
        Returns:
            List of Data objects with features [c, p, u, v]
        """
        graphs = []
        for time_dir in time_dirs:
            try:
                graph = self.build_graph(
                    time_dir, fields, 
                    adaptive_sampling=adaptive_sampling,
                    radius=radius,
                    include_concentration=include_concentration
                )
                graphs.append(graph)
            except Exception as e:
                print(f"Warning: Could not build graph for {time_dir}: {e}")
                continue
        
        return graphs

