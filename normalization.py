"""
Data normalization utilities for improving model training.
"""
import numpy as np
import torch
from typing import Tuple, Dict


class FeatureNormalizer:
    """
    Normalize features using statistics computed from training data.
    Supports both standardization and min-max normalization.
    Can normalize fields separately (e.g., velocity and pressure separately).
    """
    
    def __init__(self, method='standardize', normalize_per_field=False):
        """
        Initialize normalizer.
        
        Args:
            method: 'standardize' (zero mean, unit std) or 'minmax' (0-1 range)
            normalize_per_field: If True, normalize each field separately (better for mixed scales)
        """
        self.method = method
        self.normalize_per_field = normalize_per_field
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
        self.fitted = False
    
    def fit(self, features: np.ndarray):
        """
        Compute normalization statistics from data.
        
        Args:
            features: Array of shape (n_samples, n_features) or list of arrays
        """
        if isinstance(features, list):
            # Concatenate all features
            all_features = np.concatenate(features, axis=0)
        else:
            all_features = features
        
        if self.normalize_per_field:
            # Normalize each field separately (better for mixed scales like velocity + pressure)
            n_features = all_features.shape[1]
            if self.method == 'standardize':
                self.mean = np.zeros((1, n_features))
                self.std = np.ones((1, n_features))
                for i in range(n_features):
                    field_data = all_features[:, i]
                    self.mean[0, i] = np.mean(field_data)
                    self.std[0, i] = np.std(field_data)
                    if self.std[0, i] < 1e-8:
                        self.std[0, i] = 1.0
            elif self.method == 'minmax':
                self.min = np.zeros((1, n_features))
                self.max = np.ones((1, n_features))
                for i in range(n_features):
                    field_data = all_features[:, i]
                    self.min[0, i] = np.min(field_data)
                    self.max[0, i] = np.max(field_data)
                self.range = self.max - self.min
                self.range = np.where(self.range < 1e-8, 1.0, self.range)
        else:
            # Normalize all features together
            if self.method == 'standardize':
                self.mean = np.mean(all_features, axis=0, keepdims=True)
                self.std = np.std(all_features, axis=0, keepdims=True)
                # Avoid division by zero
                self.std = np.where(self.std < 1e-8, 1.0, self.std)
            elif self.method == 'minmax':
                self.min = np.min(all_features, axis=0, keepdims=True)
                self.max = np.max(all_features, axis=0, keepdims=True)
                # Avoid division by zero
                self.range = self.max - self.min
                self.range = np.where(self.range < 1e-8, 1.0, self.range)
            else:
                raise ValueError(f"Unknown normalization method: {self.method}")
        
        self.fitted = True
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features.
        
        Args:
            features: Array of shape (n_samples, n_features)
        
        Returns:
            Normalized features
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform")
        
        if self.method == 'standardize':
            return (features - self.mean) / self.std
        elif self.method == 'minmax':
            return (features - self.min) / self.range
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
    
    def inverse_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Denormalize features.
        
        Args:
            features: Normalized features
        
        Returns:
            Original scale features
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before inverse_transform")
        
        if self.method == 'standardize':
            return features * self.std + self.mean
        elif self.method == 'minmax':
            return features * self.range + self.min
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
    
    def get_stats(self) -> Dict:
        """Get normalization statistics."""
        if not self.fitted:
            return {}
        
        if self.method == 'standardize':
            return {'mean': self.mean, 'std': self.std}
        else:
            return {'min': self.min, 'max': self.max}


def normalize_graph_features(graphs, normalizer=None, fit=True):
    """
    Normalize features in a list of graphs.
    
    Args:
        graphs: List of PyTorch Geometric Data objects
        normalizer: FeatureNormalizer instance (if None, creates new one)
        fit: If True, fit normalizer on provided graphs
    
    Returns:
        Normalized graphs, normalizer
    """
    if normalizer is None:
        normalizer = FeatureNormalizer(method='standardize')
    
    # Extract all features for fitting
    if fit:
        all_features = [g.x.numpy() for g in graphs]
        normalizer.fit(all_features)
    
    # Normalize each graph
    normalized_graphs = []
    for graph in graphs:
        normalized_x = normalizer.transform(graph.x.numpy())
        normalized_graph = type(graph)(
            x=torch.tensor(normalized_x, dtype=torch.float32),
            edge_index=graph.edge_index,
            num_nodes=graph.num_nodes
        )
        normalized_graphs.append(normalized_graph)
    
    return normalized_graphs, normalizer

