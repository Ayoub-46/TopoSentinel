from collections import deque
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import copy
import warnings
# --- Add scipy for distance matrix calculation ---
from scipy.spatial.distance import pdist, squareform

# Import necessary components from your framework
from ..fl.server import FedAvgAggregator
from .utils import flatten_weights, unflatten_weights, reduce_dimension_pca, reduce_dimension_dct
from ..persistent_homology.analyzer import TopologicalAnalyser
from ..persistent_homology.metrics import magnitude_cosine_distance

# Import persim for bottleneck distance
try:
    import persim
except ImportError:
    print("Error: persim library not found. Please install using 'pip install persim'")
    persim = None # Allow import but fail later if used


def calculate_mad_outliers(data: np.ndarray, threshold: float = 2.0) -> Tuple[np.ndarray, float, float]:
    """Identifies outliers using the Median Absolute Deviation (MAD) method."""
    if data.ndim != 1 or len(data) == 0:
        return np.array([False] * len(data)), np.nan, np.nan
    with warnings.catch_warnings():
         warnings.simplefilter("ignore", category=RuntimeWarning)
         data_median = np.median(data)
         abs_dev = np.abs(data - data_median)
         mad = np.median(abs_dev)
    if mad < 1e-9:
        modified_z_score = np.where(np.abs(data - data_median) > 1e-9, threshold + 1, 0)
    else:
        # 1.4826 = 1 / inverse_cdf(0.75) for standard normal -> makes MAD comparable to std dev
        modified_z_score = abs_dev / (mad * 1.4826) if mad > 1e-9 else np.inf * np.sign(abs_dev)

    is_outlier = modified_z_score > threshold
    return is_outlier, data_median, mad

class TopologicalDefenseServer(FedAvgAggregator):
    """
    Server implementing topological defense with Decaying Bottleneck Threshold
    and NN-MAD filtering. Prints key defense info to console.
    """
    def __init__(self, model: torch.nn.Module, testloader=None, device: Optional[torch.device]=None, defense_config: Optional[Dict] = None, **kwargs):
        super().__init__(model, testloader, device) 
        if defense_config is None: defense_config = {}
        if persim is None: raise ImportError("persim library is required but not installed.")

        # --- Standard Params ---
        self.reduction_method = defense_config.get('reduction_method', 'pca').lower()
        self.pca_variance_ratio = defense_config.get('pca_variance_ratio', 0.95)
        self.dct_components = defense_config.get('dct_components', 50)
        self.metric_alpha = defense_config.get('metric_alpha', 0.5)
        self.mad_threshold = defense_config.get('mad_threshold', 2.0)
        self.min_clients_for_defense = defense_config.get('min_clients_for_defense', 3)

        # --- Decaying Bottleneck Threshold Params ---
        self.bottleneck_initial_threshold = defense_config.get('bottleneck_initial_threshold', 1.0)
        self.bottleneck_decay_rate = defense_config.get('bottleneck_decay_rate', 0.99)
        self.bottleneck_min_threshold = defense_config.get('bottleneck_min_threshold', 0.01)

        print(f"--- Initializing TopologicalDefenseServer (Decaying Bottleneck) ---")
        print(f" Reduction: {self.reduction_method}, Metric Alpha: {self.metric_alpha}")
        print(f" Decaying Bottleneck: Initial={self.bottleneck_initial_threshold}, DecayRate={self.bottleneck_decay_rate}, Min={self.bottleneck_min_threshold}")
        print(f" Filtering Metric: Nearest Neighbor magnitude_cosine")
        print(f" MAD Threshold: {self.mad_threshold}")
        print(f" Min Clients: {self.min_clients_for_defense}")
        print(f"-----------------------------------------")

        self.topological_analyser = TopologicalAnalyser(
            homology_dimensions=(0,),
            metric=magnitude_cosine_distance,
            metric_params={'alpha': self.metric_alpha}
        )
        self.prev_diagram: Optional[np.ndarray] = None
        self.round_counter = 0 

    def aggregate(self) -> Dict[str, torch.Tensor]:
        num_received = len(self.received_params)
        current_round_print = self.round_counter + 1
        print(f"\n--- TopoDefense Aggregation (Round {current_round_print}, Received: {num_received}) ---")

        if num_received < self.min_clients_for_defense:
            print(f" Updates < min clients. Falling back to FedAvg.")
            aggregated_params = super().aggregate() 
            self.prev_diagram = None
            self.round_counter += 1
            return aggregated_params

        perform_filtering = False
        indices_to_keep = list(range(num_received))
        reduced_updates = None

        current_bottleneck_threshold = max(
            self.bottleneck_min_threshold,
            self.bottleneck_initial_threshold * (self.bottleneck_decay_rate ** self.round_counter)
        )
        print(f" Current Decaying Bottleneck Threshold: {current_bottleneck_threshold:.4f}")

        try:
            # 1. Flatten Deltas
            current_global_params = self.get_params()
            client_deltas = []; metadata = None
            for client_params in self.received_params:
                delta = {n: client_params[n].cpu() - current_global_params[n].cpu() for n in client_params if n in current_global_params}
                flat_delta, meta = flatten_weights(delta)
                if metadata is None: metadata = meta
                client_deltas.append(flat_delta)
            if not client_deltas:
                 self.round_counter += 1; return super().aggregate() # Fallback resets buffers
            flat_updates_matrix = np.vstack(client_deltas).astype(np.float64)

            # 2. Reduce Dimension
            if self.reduction_method == 'pca': reduced_updates = reduce_dimension_pca(flat_updates_matrix, self.pca_variance_ratio)
            elif self.reduction_method == 'dct': reduced_updates = reduce_dimension_dct(flat_updates_matrix, self.dct_components)
            else: reduced_updates = flat_updates_matrix
            if reduced_updates is None or reduced_updates.shape[0] != num_received:
                 self.round_counter += 1; return super().aggregate() # Fallback resets buffers

            # 3. Compute H0 Diagram
            current_diagram = self.topological_analyser.compute_diagram(reduced_updates)
            if current_diagram is None or current_diagram.shape[0] == 0:
                 self.prev_diagram = None; self.round_counter += 1; return super().aggregate() # Fallback resets buffers
            h0_diagram = current_diagram[current_diagram[:, 2] == 0]

            # 4. Inter-round Analysis: Decaying Threshold & Trigger
            bottleneck_dist = np.nan
            if self.prev_diagram is not None and self.prev_diagram.shape[0] > 0 and h0_diagram.shape[0] > 0:
                finite_prev_h0 = self.prev_diagram[np.isfinite(self.prev_diagram[:, 1])]
                finite_curr_h0 = h0_diagram[np.isfinite(h0_diagram[:, 1])]
                if finite_prev_h0.shape[0] > 0 and finite_curr_h0.shape[0] > 0:
                     try:
                         bottleneck_dist = persim.bottleneck(finite_prev_h0[:, :2], finite_curr_h0[:, :2])
                         print(f" Bottleneck distance to previous round: {bottleneck_dist:.4f}")
                     except Exception as persim_err:
                         print(f"Warning: persim.bottleneck calculation failed: {persim_err}")
                         bottleneck_dist = np.nan

            if not np.isnan(bottleneck_dist) and bottleneck_dist > current_bottleneck_threshold:
                perform_filtering = True
                print(f" *** Change detected (Dist {bottleneck_dist:.4f} > Threshold {current_bottleneck_threshold:.4f}). Triggering NN-MAD filtering. ***")
            else:
                print(" No significant change detected or distance invalid. Updating prev_diagram.")
                self.prev_diagram = h0_diagram 

            # 5. Intra-round Filtering (NN-MAD)
            is_outlier_flags = np.array([False] * num_received)
            if perform_filtering and reduced_updates is not None and reduced_updates.shape[0] > 1:
                print(f"--- Filtering based on MAD (Threshold={self.mad_threshold}) of NN distances ---")
                metric_func = lambda u, v: magnitude_cosine_distance(u, v, alpha=self.metric_alpha)
                try:
                    pairwise_distances_matrix = squareform(pdist(reduced_updates, metric=metric_func))
                    np.fill_diagonal(pairwise_distances_matrix, np.inf)
                    nearest_neighbor_distances = np.min(pairwise_distances_matrix, axis=1)

                    is_outlier_flags, nn_dist_median, nn_dist_mad = calculate_mad_outliers(
                        nearest_neighbor_distances, threshold=self.mad_threshold
                    )
                    outlier_indices = np.where(is_outlier_flags)[0]
                    indices_to_keep = np.where(~is_outlier_flags)[0].tolist()

                    num_filtered = len(outlier_indices)
                    print(f" NN Distance Median: {nn_dist_median:.4f}, MAD: {nn_dist_mad:.4f}")
                    if num_filtered > 0:
                        print(f" Identified {num_filtered} outliers (Indices: {outlier_indices.tolist()}). Keeping {len(indices_to_keep)}.")
                        if len(indices_to_keep) == 0:
                            print("Warning: All clients flagged as outliers. Keeping all.")
                            indices_to_keep = list(range(num_received))
                            is_outlier_flags[:] = False 
                    else:
                        print(" No outliers detected by NN Dist MAD.")
                except Exception as dist_err:
                     print(f"Error calculating distances: {dist_err}. Keeping all clients.")
                     indices_to_keep = list(range(num_received))
                     is_outlier_flags[:] = False 
            elif perform_filtering:
                 print(" Skipping filtering: Not enough points.")

        except Exception as e:
            print(f"!!! Error during topological defense steps: {e}. Falling back to FedAvg. !!!")
            warnings.warn(f"Topological Defense Error: {e}", RuntimeWarning)
            self.prev_diagram = None
            indices_to_keep = list(range(num_received)) 

        # 6. Aggregate Kept Clients
        if len(indices_to_keep) < num_received: print(f" Aggregating {len(indices_to_keep)} clients.")
        else: print(f" Aggregating all {num_received} received clients.")

        params_to_aggregate = [self.received_params[i] for i in indices_to_keep] 
        lens_to_aggregate = [self.received_lens[i] for i in indices_to_keep]
        if not params_to_aggregate:
            print("Warning: No clients left. Skipping aggregation.")
            self.prev_diagram = None
            aggregated_params = self.get_params() 
            self.received_params = []; self.received_lens = []
        else:
            aggregated_params = {}
            first = params_to_aggregate[0]
            for k in first.keys():
                 acc = torch.zeros_like(first[k], dtype=torch.float32, device='cpu')
                 for i, client_params in enumerate(params_to_aggregate):
                     weight = float(lens_to_aggregate[i]) / float(sum(lens_to_aggregate))
                     acc += client_params[k].cpu().float() * weight
                 aggregated_params[k] = acc.to(self.device)
            
            self.set_params({k: v.to(self.device) for k, v in aggregated_params.items()})
            self.received_params = []; self.received_lens = []


        print(f"--- Topological Defense Aggregation Finished ---")
        self.round_counter += 1 
        return aggregated_params