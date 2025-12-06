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
    
    def build_edge_index(self, adaptive_sampling=False, radius=None, 
                        cell_centers=None, gradient_info=None) -> torch.Tensor:
        """
        Build edge index from mesh connectivity.
        
        Args:
            adaptive_sampling: If True, use adaptive spatial sampling based on radius
            radius: Sampling radius for adaptive method (if None, uses mesh-based connectivity)
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
        # Internal faces create edges between owner and neighbour cells
        # Create bidirectional edges
        edges = []
        
        # Forward edges (owner -> neighbour)
        for i in range(len(self.neighbour)):
            owner_cell = self.owner[i]
            neighbour_cell = self.neighbour[i]
            edges.append([owner_cell, neighbour_cell])
            edges.append([neighbour_cell, owner_cell])  # Bidirectional
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Remove duplicate edges (if any)
        edge_index = torch.unique(edge_index, dim=1)
        
        return edge_index
    
    def _build_adaptive_edge_index(self, cell_centers, radius=None, gradient_info=None) -> torch.Tensor:
        """
        Build edges using adaptive spatial sampling.
        
        Adaptive sampling formula: ne[i] = {j | (x_j - x_i)^2 + (y_j - y_i)^2 < r^2}
        
        This method helps capture multi-scale gradient information, especially in regions
        with large gradient changes (near walls, obstacles). By sampling neighbors within
        a spatial radius, we ensure the neural network can perceive drastic gradient changes.
        
        Args:
            cell_centers: Cell center coordinates [n_cells, 3]
            radius: Sampling radius (if None, computed adaptively)
            gradient_info: Optional gradient information for adaptive radius
        """
        n_cells = len(cell_centers)
        edges = []
        
        # Compute adaptive radius if not provided
        if radius is None:
            # Compute average distance to nearest neighbors from mesh connectivity
            mesh_edges = self._build_mesh_based_edge_index()
            mesh_distances = []
            
            # Sample a subset for efficiency
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
                # Use 1.5x the median distance as radius
                # This ensures we capture more neighbors in regions with large gradients
                radius = np.median(mesh_distances) * 1.5
            else:
                # Fallback: use a fraction of domain size
                domain_size = np.linalg.norm(
                    cell_centers.max(axis=0) - cell_centers.min(axis=0)
                )
                radius = domain_size * 0.1
        
        print(f"Using adaptive sampling with radius: {radius:.6f}")
        
        # For each cell, find neighbors within radius
        # Use 2D distance (x, y) for 2D flow visualization
        cell_centers_2d = cell_centers[:, :2]  # Use only x, y coordinates
        
        # Use KDTree for efficient spatial queries
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(cell_centers_2d)
            # Find all pairs within radius
            pairs = tree.query_pairs(radius, output_type='ndarray')
            
            # Create bidirectional edges
            for i, j in pairs:
                edges.append([int(i), int(j)])
                edges.append([int(j), int(i)])
        except ImportError:
            # Fallback: compute distances directly (slower but works without scipy)
            print("Warning: scipy not available, using slower distance computation")
            for i in range(n_cells):
                distances = np.linalg.norm(
                    cell_centers_2d - cell_centers_2d[i], axis=1
                )
                neighbors = np.where((distances < radius) & (distances > 0))[0]
                for j in neighbors:
                    edges.append([i, int(j)])
                    edges.append([int(j), i])
        
        # Also keep original mesh connectivity for structure
        # This ensures we don't lose the mesh topology
        mesh_edges = self._build_mesh_based_edge_index()
        mesh_edges_list = mesh_edges.t().tolist()
        edges.extend(mesh_edges_list)
        
        # Remove duplicates
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index = torch.unique(edge_index, dim=1)
        
        mesh_edge_count = mesh_edges.shape[1]
        adaptive_edge_count = edge_index.shape[1]
        print(f"Adaptive sampling: {adaptive_edge_count} edges (mesh-based: {mesh_edge_count} edges, "
              f"added {adaptive_edge_count - mesh_edge_count} spatial edges)")
        
        return edge_index
    
    def compute_cell_features(self, time_dir: str, 
                             fields: Optional[Dict[str, str]] = None,
                             include_coordinates: bool = False) -> torch.Tensor:
        """
        Compute node features from field data.
        
        Args:
            time_dir: Time directory name
            fields: Dictionary mapping feature names to field names
                   Default: {'velocity': 'U', 'pressure': 'p'}
            include_coordinates: If True, embed cell center coordinates in features
        
        Returns:
            Feature tensor of shape (n_cells, n_features)
        """
        if fields is None:
            fields = {'velocity': 'U', 'pressure': 'p'}
        
        features = []
        
        # Parse velocity (3D vector -> 3 features, but for 2D flow use only u, v)
        try:
            velocity = self.parser.parse_vector_field(time_dir, fields.get('velocity', 'U'))
            # Ensure it's 2D
            if velocity.ndim == 1:
                velocity = velocity.reshape(-1, 1)
            elif velocity.ndim == 2 and velocity.shape[1] != 3:
                # If it's 2D but wrong shape, reshape
                velocity = velocity.reshape(-1, 3)
            # For 2D flow, use only u (x-direction) and v (y-direction)
            # Keep all 3 components but note that w (z-direction) should be ~0 for 2D
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
        
        # Embed cell center coordinates (implicit coordinate information)
        if include_coordinates:
            cell_centers = self.parser.compute_cell_centers()
            # Normalize coordinates to [0, 1] range for better learning
            coords = cell_centers.copy()
            coords_min = coords.min(axis=0)
            coords_max = coords.max(axis=0)
            coords_range = coords_max - coords_min
            coords_range[coords_range == 0] = 1  # Avoid division by zero
            coords_normalized = (coords - coords_min) / coords_range
            features.append(coords_normalized)
        
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
                   target_fields: Optional[Dict[str, str]] = None,
                   adaptive_sampling: bool = False,
                   radius: Optional[float] = None,
                   include_coordinates: bool = False) -> Data:
        """
        Build a PyTorch Geometric Data object from OpenFOAM data.
        
        Args:
            time_dir: Time directory name
            fields: Dictionary of input field names
            target_fields: Dictionary of target field names (for supervised learning)
            adaptive_sampling: If True, use adaptive spatial sampling for edges
            radius: Sampling radius for adaptive method (if None, computed automatically)
        
        Returns:
            PyTorch Geometric Data object
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
        
        # Build node features
        x = self.compute_cell_features(time_dir, fields, include_coordinates=include_coordinates)
        
        # Build target features (if provided)
        y = None
        if target_fields:
            y = self.compute_cell_features(time_dir, target_fields, include_coordinates=False)
        
        # Create graph data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            num_nodes=self.n_cells
        )
        
        return data
    
    def build_temporal_graphs(self, time_dirs: List[str],
                             fields: Optional[Dict[str, str]] = None,
                             adaptive_sampling: bool = False,
                             radius: Optional[float] = None) -> list:
        """
        Build graphs for multiple time steps.
        
        Args:
            time_dirs: List of time directory names
            fields: Dictionary of field names
            adaptive_sampling: If True, use adaptive spatial sampling for edges
            radius: Sampling radius for adaptive method
        
        Returns:
            List of Data objects
        """
        graphs = []
        for time_dir in time_dirs:
            try:
                graph = self.build_graph(time_dir, fields, 
                                       adaptive_sampling=adaptive_sampling,
                                       radius=radius)
                graphs.append(graph)
            except Exception as e:
                print(f"Warning: Could not build graph for {time_dir}: {e}")
                continue
        
        return graphs

