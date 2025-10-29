import numpy as np
from gtda.homology import VietorisRipsPersistence
from typing import Callable, Optional, Tuple

class TopologicalAnalyser:
    """
    Computes persistence diagrams using Vietoris-Rips filtration.
    Can use standard metrics or a custom distance function/matrix.
    """
    def __init__(self,
                 homology_dimensions: Tuple[int, ...] = (0,),
                 metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
                 metric_params: Optional[dict] = None,
                 n_jobs: Optional[int] = -1):
        """
        Initializes the topological analyser.

        Args:
            homology_dimensions (tuple, optional): Dimensions (e.g., (0,) for connected components,
                                                 (0, 1) for components and loops). Defaults to (0,).
            metric (Callable, optional): A custom distance function mapping
                                          (point1_vector, point2_vector) -> distance.
                                          If None, uses default 'euclidean'. Defaults to None.
                                          Can also be a precomputed distance matrix (set metric='precomputed').
            metric_params (dict, optional): Additional parameters for the metric function
                                           (e.g., {'alpha': 0.7} for magnitude_cosine_distance).
                                           Defaults to None.
            n_jobs (int, optional): Number of jobs to use for computation (-1 means all CPUs).
                                    Defaults to -1.
        """
           
        # If a custom callable metric is provided, wrap it if it needs extra params
        if callable(metric):
            if metric_params:
                # Wrap the metric to include the parameters
                def custom_metric_wrapper(p1, p2):
                    return metric(p1, p2, **metric_params)
                metric_arg = custom_metric_wrapper
            else:
                metric_arg = metric
            self.persistence = VietorisRipsPersistence(
                metric=metric_arg,
                homology_dimensions=homology_dimensions,
                n_jobs=n_jobs
            )
        else:
            metric_arg = metric if isinstance(metric, str) else 'euclidean'
            self.persistence = VietorisRipsPersistence(
                metric=metric_arg,
                homology_dimensions=homology_dimensions,
                metric_params=metric_params if metric_arg != 'precomputed' else None, # Only pass if not precomputed
                n_jobs=n_jobs
            )
        
        self.metric_used = metric_arg

    def compute_diagram(self, data: np.ndarray) -> np.ndarray:
        """
        Computes the persistence diagram for the given data (point cloud or distance matrix).

        Args:
            data (np.ndarray): A NumPy array representing the data.
                                - If using a standard or custom metric function:
                                  Shape should be (n_samples, n_features).
                                - If metric='precomputed':
                                  Shape should be (n_samples, n_samples) representing the distance matrix.

        Returns:
            np.ndarray: The persistence diagram, shape (n_features_in_diagram, 3).
                        Columns are [birth, death, dimension]. Returns empty array if input is invalid.
        """
        # --- Input Validation ---
        if not isinstance(data, np.ndarray):
            print("Warning: Input data must be a NumPy array.")
            return np.empty((0, 3))

        if self.metric_used == 'precomputed':
            if data.ndim != 2 or data.shape[0] != data.shape[1]:
                print(f"Warning: Metric is 'precomputed', expected square distance matrix, but got shape {data.shape}.")
                return np.empty((0, 3))
            n_samples = data.shape[0]
        else:
            if data.ndim != 2:
                print(f"Warning: Expected 2D data (n_samples, n_features), but got shape {data.shape}.")
                return np.empty((0, 3))
            n_samples = data.shape[0]

        if n_samples < 2: 
             return np.empty((0, 3)) # Return empty diagram

        # --- Reshape for gtda ---
        # gtda expects input shape (n_batches, n_samples, n_features) or (n_batches, n_samples, n_samples)
        data_reshaped = data[np.newaxis, :, :]

        # --- Compute Diagram ---
        try:
            # fit_transform returns a list of diagrams (one per batch)
            diagrams_list = self.persistence.fit_transform(data_reshaped)
            # Return the diagram for the single batch
            return diagrams_list[0]
        except Exception as e:
            print(f"Error computing persistence diagram: {e}")
            # print(f"Input data shape: {data.shape}, Metric used: {self.metric_used}") # Debug info
            return np.empty((0, 3)) # Return empty on error
