"""
OpenFOAM data parser for extracting mesh and field data.
"""
import re
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional


class OpenFOAMParser:
    """Parser for OpenFOAM mesh and field data."""
    
    def __init__(self, data_dir: str):
        """
        Initialize parser with OpenFOAM data directory.
        
        Args:
            data_dir: Path to OpenFOAM case directory
        """
        self.data_dir = Path(data_dir)
        self.mesh_dir = self.data_dir / "constant" / "polyMesh"
        
    def parse_points(self) -> np.ndarray:
        """Parse mesh points (coordinates)."""
        points_file = self.mesh_dir / "points"
        with open(points_file, 'r') as f:
            content = f.read()
        
        # Extract number of points
        match = re.search(r'(\d+)\s*\(', content)
        if not match:
            raise ValueError("Could not find number of points")
        n_points = int(match.group(1))
        
        # Extract point coordinates - find all coordinate tuples
        # Pattern matches: (x y z) where x, y, z are numbers
        coord_pattern = r'\(([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\)'
        matches = re.findall(coord_pattern, content)
        
        points = []
        for match in matches:
            coords = [float(x) for x in match]
            points.append(coords)
        
        if len(points) != n_points:
            # Fallback: try simpler pattern
            coord_pattern = r'\(([^)]+)\)'
            matches = re.findall(coord_pattern, content)
            points = []
            for match in matches:
                # Skip if it's the outer parentheses
                if match.count('(') > 0 or match.count(')') > 0:
                    continue
                try:
                    coords = [float(x) for x in match.split()]
                    if len(coords) == 3:
                        points.append(coords)
                except ValueError:
                    continue
        
        return np.array(points)
    
    def parse_faces(self) -> List[List[int]]:
        """
        Parse mesh faces (list of point indices for each face).
        
        Returns:
            List of faces, where each face is a list of point indices
        """
        faces_file = self.mesh_dir / "faces"
        with open(faces_file, 'r') as f:
            content = f.read()
        
        # Extract number of faces
        match = re.search(r'(\d+)\s*\(', content)
        if not match:
            raise ValueError("Could not find number of faces")
        n_faces = int(match.group(1))
        
        # Extract face definitions
        # Pattern: nPoints(point1 point2 ... pointN)
        face_pattern = r'(\d+)\(([^)]+)\)'
        matches = re.findall(face_pattern, content)
        
        faces = []
        for n_points_str, points_str in matches:
            n_points = int(n_points_str)
            point_indices = [int(x) for x in points_str.split() if x.strip().isdigit()]
            # OpenFOAM uses 1-based indexing, convert to 0-based
            point_indices = [idx - 1 for idx in point_indices]
            if len(point_indices) == n_points:
                faces.append(point_indices)
            else:
                # Handle case where points might be on multiple lines
                # Try to get all points
                if len(point_indices) > 0:
                    faces.append(point_indices[:n_points] if len(point_indices) >= n_points else point_indices)
        
        return faces
    
    def parse_connectivity(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse mesh connectivity (owner and neighbour arrays).
        
        Returns:
            owner: Array of cell indices owning each face
            neighbour: Array of cell indices neighbouring each face (internal faces only)
        """
        # Parse owner
        owner_file = self.mesh_dir / "owner"
        with open(owner_file, 'r') as f:
            content = f.read()
        
        match = re.search(r'(\d+)\s*\(', content)
        n_faces = int(match.group(1))
        
        match = re.search(r'\(([^)]+)\)', content)
        owner_str = match.group(1)
        owner = np.array([int(x) for x in owner_str.split() if x.strip().isdigit()])
        
        # Parse neighbour (only internal faces)
        neighbour_file = self.mesh_dir / "neighbour"
        with open(neighbour_file, 'r') as f:
            content = f.read()
        
        match = re.search(r'(\d+)\s*\(', content)
        n_internal = int(match.group(1))
        
        match = re.search(r'\(([^)]+)\)', content)
        neighbour_str = match.group(1)
        neighbour = np.array([int(x) for x in neighbour_str.split() if x.strip().isdigit()])
        
        return owner, neighbour
    
    def compute_cell_centers(self) -> np.ndarray:
        """
        Compute cell centers from mesh geometry.
        For each cell, collect all points from faces belonging to that cell
        and compute the centroid.
        """
        points = self.parse_points()
        owner, neighbour = self.parse_connectivity()
        faces = self.parse_faces()
        
        n_cells = max(owner.max(), neighbour.max()) + 1
        cell_centers = np.zeros((n_cells, 3))
        
        # For each cell, collect all points from its faces
        cell_points = [[] for _ in range(n_cells)]
        
        # Process internal faces (have both owner and neighbour)
        for i in range(len(neighbour)):
            face_idx = i
            owner_cell = owner[face_idx]
            neighbour_cell = neighbour[i]
            
            # Add face points to both owner and neighbour cells
            if face_idx < len(faces):
                face_point_indices = faces[face_idx]
                for point_idx in face_point_indices:
                    if point_idx < len(points):
                        cell_points[owner_cell].append(points[point_idx])
                        cell_points[neighbour_cell].append(points[point_idx])
        
        # Process boundary faces (only have owner)
        n_internal = len(neighbour)
        for i in range(n_internal, len(owner)):
            face_idx = i
            owner_cell = owner[face_idx]
            
            if face_idx < len(faces):
                face_point_indices = faces[face_idx]
                for point_idx in face_point_indices:
                    if point_idx < len(points):
                        cell_points[owner_cell].append(points[point_idx])
        
        # Compute cell centers as centroids of collected points
        for cell_idx in range(n_cells):
            if len(cell_points[cell_idx]) > 0:
                # Remove duplicates and compute centroid
                unique_points = np.unique(np.array(cell_points[cell_idx]), axis=0)
                cell_centers[cell_idx] = np.mean(unique_points, axis=0)
            else:
                # Fallback: use zero (shouldn't happen)
                cell_centers[cell_idx] = np.array([0.0, 0.0, 0.0])
        
        return cell_centers
    
    def parse_vector_field(self, time_dir: str, field_name: str) -> np.ndarray:
        """
        Parse a vector field (e.g., velocity U).
        
        Args:
            time_dir: Time directory name (e.g., "0.001")
            field_name: Field name (e.g., "U")
        
        Returns:
            Array of shape (n_cells, 3) with vector values
        """
        field_file = self.data_dir / time_dir / field_name
        if not field_file.exists():
            raise FileNotFoundError(f"Field file not found: {field_file}")
        
        with open(field_file, 'r') as f:
            content = f.read()
        
        # Check if field is uniform (initial conditions)
        uniform_match = re.search(r'internalField\s+uniform\s+\(([^)]+)\)', content)
        if uniform_match:
            # Get number of cells from mesh
            owner, neighbour = self.parse_connectivity()
            n_cells = max(owner.max(), neighbour.max()) + 1
            # Parse uniform value
            uniform_str = uniform_match.group(1)
            uniform_value = [float(x) for x in uniform_str.split()]
            # Return array with same value for all cells
            return np.tile(uniform_value, (n_cells, 1))
        
        # Extract number of cells - handle newlines between List<vector> and number
        match = re.search(r'internalField\s+nonuniform\s+List<vector>\s*\n\s*(\d+)', content)
        if not match:
            # Try without newline
            match = re.search(r'internalField\s+nonuniform\s+List<vector>\s+(\d+)', content)
        if not match:
            raise ValueError(f"Could not find internalField in {field_name}")
        n_cells = int(match.group(1))
        
        # Extract vector values - find the content between the outer parentheses
        # Need to find the opening ( after the number and match until the closing )
        # Pattern: internalField nonuniform List<vector> \n number \n ( ... )
        # Find the position after the number
        match_start = re.search(r'internalField\s+nonuniform\s+List<vector>\s*\n\s*\d+\s*\n\s*\(', content, re.DOTALL)
        if not match_start:
            match_start = re.search(r'internalField\s+nonuniform\s+List<vector>\s+\d+\s*\(', content, re.DOTALL)
        if not match_start:
            raise ValueError(f"Could not find vector values start in {field_name}")
        
        # Find matching closing parenthesis (count parentheses)
        start_pos = match_start.end() - 1  # Position of opening (
        depth = 0
        end_pos = start_pos
        
        for i in range(start_pos, len(content)):
            if content[i] == '(':
                depth += 1
            elif content[i] == ')':
                depth -= 1
                if depth == 0:
                    end_pos = i
                    break
        
        values_str = content[start_pos + 1:end_pos]  # Content between parentheses
        
        # Extract all vector tuples - pattern: (x y z)
        # Use simpler pattern that matches parentheses with content
        vector_pattern = r'\(([^)]+)\)'
        matches = re.findall(vector_pattern, values_str)
        
        vectors = []
        for match in matches:
            try:
                coords = [float(x) for x in match.split()]
                if len(coords) == 3:
                    vectors.append(coords)
            except ValueError:
                continue
        
        if len(vectors) != n_cells:
            raise ValueError(f"Expected {n_cells} vectors but found {len(vectors)}")
        
        return np.array(vectors)
    
    def parse_scalar_field(self, time_dir: str, field_name: str) -> np.ndarray:
        """
        Parse a scalar field (e.g., pressure p).
        
        Args:
            time_dir: Time directory name (e.g., "0.001")
            field_name: Field name (e.g., "p")
        
        Returns:
            Array of shape (n_cells,) with scalar values
        """
        field_file = self.data_dir / time_dir / field_name
        if not field_file.exists():
            raise FileNotFoundError(f"Field file not found: {field_file}")
        
        with open(field_file, 'r') as f:
            content = f.read()
        
        # Check if field is uniform (initial conditions)
        uniform_match = re.search(r'internalField\s+uniform\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', content)
        if uniform_match:
            # Get number of cells from mesh
            owner, neighbour = self.parse_connectivity()
            n_cells = max(owner.max(), neighbour.max()) + 1
            # Parse uniform value
            uniform_value = float(uniform_match.group(1))
            # Return array with same value for all cells
            return np.full(n_cells, uniform_value)
        
        # Extract number of cells - handle newlines
        match = re.search(r'internalField\s+nonuniform\s+List<scalar>\s*\n\s*(\d+)', content)
        if not match:
            # Try without newline
            match = re.search(r'internalField\s+nonuniform\s+List<scalar>\s+(\d+)', content)
        if not match:
            raise ValueError(f"Could not find internalField in {field_name}")
        n_cells = int(match.group(1))
        
        # Extract scalar values - find the content between the outer parentheses
        # Find the position after the number
        match_start = re.search(r'internalField\s+nonuniform\s+List<scalar>\s*\n\s*\d+\s*\n\s*\(', content, re.DOTALL)
        if not match_start:
            match_start = re.search(r'internalField\s+nonuniform\s+List<scalar>\s+\d+\s*\(', content, re.DOTALL)
        if not match_start:
            raise ValueError(f"Could not find scalar values start in {field_name}")
        
        # Find matching closing parenthesis (count parentheses)
        start_pos = match_start.end() - 1  # Position of opening (
        depth = 0
        end_pos = start_pos
        
        for i in range(start_pos, len(content)):
            if content[i] == '(':
                depth += 1
            elif content[i] == ')':
                depth -= 1
                if depth == 0:
                    end_pos = i
                    break
        
        values_str = content[start_pos + 1:end_pos]  # Content between parentheses
        
        # Extract all scalar values (they're space-separated, may have newlines)
        # Pattern to match numbers (including scientific notation)
        number_pattern = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
        matches = re.findall(number_pattern, values_str)
        
        values = []
        for match in matches:
            try:
                values.append(float(match))
            except ValueError:
                continue
        
        if len(values) != n_cells:
            raise ValueError(f"Expected {n_cells} scalar values but found {len(values)}")
        
        return np.array(values)
    
    def get_time_directories(self) -> List[str]:
        """Get all time directories sorted by time value."""
        time_dirs = []
        for item in self.data_dir.iterdir():
            if item.is_dir() and item.name.replace('.', '').replace('-', '').isdigit():
                try:
                    time_val = float(item.name)
                    time_dirs.append((time_val, item.name))
                except ValueError:
                    continue
        
        time_dirs.sort(key=lambda x: x[0])
        return [name for _, name in time_dirs]
    
    def get_cell_centers(self) -> np.ndarray:
        """
        Compute cell centers from mesh connectivity.
        This is a simplified version - in practice, you'd compute from face centers.
        For now, we'll use a centroid approximation.
        """
        points = self.parse_points()
        owner, neighbour = self.parse_connectivity()
        
        # Get number of cells (max cell index + 1)
        n_cells = max(owner.max(), neighbour.max()) + 1
        
        # Simple approximation: use average of connected points
        # In practice, OpenFOAM stores cell centers, but we'll compute from points
        cell_centers = np.zeros((n_cells, 3))
        cell_point_counts = np.zeros(n_cells)
        
        # This is simplified - proper implementation would use face centers
        # For now, we'll use a bounding box approach
        for i, point in enumerate(points):
            # Find which cell this point belongs to (simplified)
            # In practice, need proper cell-to-point mapping
            pass
        
        # For now, return zeros - will be computed properly in graph construction
        return np.zeros((n_cells, 3))

