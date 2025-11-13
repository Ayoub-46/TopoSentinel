import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import copy
import warnings
from scipy.spatial.distance import pdist, squareform, cosine
from collections import deque 

# Import necessary components
from ..fl.server import FedAvgAggregator
from .utils import flatten_weights, unflatten_weights, reduce_dimension_pca, reduce_dimension_dct
from ..persistent_homology.analyzer import TopologicalAnalyser
from ..persistent_homology.metrics import magnitude_cosine_distance # Used for TDA part
from ..persistent_homology.metrics import gaussian_kernel_distance

try:
    import persim
except ImportError:
    print("Error: persim library not found. Please install using 'pip install persim'")
    persim = None


class AnalysisServer(FedAvgAggregator): # Renamed class
    """
    Hybrid Analysis Server (Observation Mode):
    - Calculates Inter-round TDA H0 bottleneck distance (Decaying Threshold).
    - Calculates Intra-round bias vector distances from median.
    - PRINTS and STORES analysis metrics.
    - Performs NO filtering. Aggregates all clients.
    """
    def __init__(self, model: torch.nn.Module, testloader=None, device: Optional[torch.device]=None, defense_config: Optional[Dict] = None, **kwargs):
        super().__init__(model, testloader, device, **kwargs) # Pass kwargs
        if defense_config is None: defense_config = {}
        if persim is None: raise ImportError("persim library is required but not installed.")

        # TDA Parameters
        self.reduction_method = defense_config.get('reduction_method', 'pca').lower()
        self.pca_variance_ratio = defense_config.get('pca_variance_ratio', 0.95)
        self.dct_components = defense_config.get('dct_components', 50)
        self.tda_metric_alpha = defense_config.get('tda_metric_alpha', 0.5)
        self.bottleneck_initial_threshold = defense_config.get('bottleneck_initial_threshold', 0.2)
        self.bottleneck_decay_rate = defense_config.get('bottleneck_decay_rate', 0.99)
        self.bottleneck_min_threshold = defense_config.get('bottleneck_min_threshold', 0.01)

        # Bias Analysis Parameters
        self.bias_metric = defense_config.get('bias_metric', 'cosine').lower()
        # REMOVED: self.mad_threshold

        # General Params
        self.min_clients_for_defense = defense_config.get('min_clients_for_defense', 3)
        
        # --- ADDED: Attributes to store analysis results ---
        self.current_bottleneck_dist = np.nan
        self.current_bottleneck_threshold = self.bottleneck_initial_threshold
        self.current_bias_dist_stats = {'mean': np.nan, 'min': np.nan, 'max': np.nan, 'std': np.nan}
        self.malicious_ids_known = set() # For logging

        print(f"--- Initializing TopoGuardAnalysisServer (Observation Mode) ---")
        print(f" TDA Trigger: Reduction={self.reduction_method}, Metric=magnitude_cosine(alpha={self.tda_metric_alpha})")
        print(f"  Decaying Bottleneck: Initial={self.bottleneck_initial_threshold}, DecayRate={self.bottleneck_decay_rate}, Min={self.bottleneck_min_threshold}")
        print(f" Bias Analysis: Metric=DistFromMedian({self.bias_metric})")
        print(f" Min Clients: {self.min_clients_for_defense}")
        print(f"--------------------------------------------------")

        self.topological_analyser = TopologicalAnalyser(
            homology_dimensions=(0,),
            metric=gaussian_kernel_distance,
            metric_params={'gamma': self.tda_metric_alpha}
        )
        self.prev_diagram: Optional[np.ndarray] = None
        self.round_counter = 0

    def _extract_bias_vector(self, client_params: Dict[str, torch.Tensor]) -> Optional[np.ndarray]:
        """ Extracts bias parameters from a state dict and flattens them. """
        bias_tensors = []
        try:
            for name, param in client_params.items():
                if 'bias' in name:
                    if isinstance(param, torch.Tensor):
                         bias_tensors.append(param.detach().cpu().flatten())
            if not bias_tensors: return None
            return torch.cat(bias_tensors).numpy().astype(np.float64)
        except Exception as e:
            print(f"Error extracting bias vector: {e}")
            return None

    # --- ADDED: Helper to get malicious IDs from runner ---
    def set_malicious_ids(self, malicious_ids: List[int]):
         """ Call this after server init to inform it of known malicious IDs for logging. """
         self.malicious_ids_known = set(malicious_ids)
         print(f" TopoGuardAnalysisServer aware of malicious IDs: {self.malicious_ids_known}")

    def aggregate(self) -> Dict[str, torch.Tensor]:
        num_received = len(self.received_updates)
        current_round_print = self.round_counter + 1
        print(f"\n--- TopoGuard Analysis (Round {current_round_print}, Received: {num_received}) ---")

        client_ids_received = list(self.received_updates.keys())
        original_received_params = [self.received_updates[cid]['params'] for cid in client_ids_received]
        
        # --- MODIFICATION: Reset stats and never filter ---
        self.current_bottleneck_dist = np.nan
        self.current_bias_dist_stats = {'mean': np.nan, 'min': np.nan, 'max': np.nan, 'std': np.nan}
        # --- END MODIFICATION ---

        if num_received < self.min_clients_for_defense:
            print(f" Updates < min clients. Skipping analysis.")
            aggregated_params_cpu_return = super().aggregate()
            self.prev_diagram = None; self.round_counter += 1
            # Store default threshold
            self.current_bottleneck_threshold = max(
                self.bottleneck_min_threshold,
                self.bottleneck_initial_threshold * (self.bottleneck_decay_rate ** self.round_counter)
            )
            return aggregated_params_cpu_return

        current_bottleneck_threshold = max(
            self.bottleneck_min_threshold,
            self.bottleneck_initial_threshold * (self.bottleneck_decay_rate ** self.round_counter)
        )
        self.current_bottleneck_threshold = current_bottleneck_threshold # Store for logging
        print(f" Current Decaying Bottleneck Threshold: {current_bottleneck_threshold:.4f}")

        aggregated_params_cpu_return = {}

        try:
            # TDA Analysis on FULL Update Deltas (for Trigger)
            current_global_params = self.get_params()
            client_deltas_full = []; metadata = None
            for client_params in original_received_params:
                delta = {n: client_params[n].cpu() - current_global_params[n].cpu() for n in client_params if n in current_global_params}
                flat_delta, meta = flatten_weights(delta)
                if metadata is None: metadata = meta
                client_deltas_full.append(flat_delta)

            if not client_deltas_full:
                 aggregated_params_cpu_return = super().aggregate(); self.round_counter += 1; return aggregated_params_cpu_return

            flat_updates_matrix_full = np.vstack(client_deltas_full).astype(np.float64)

            if self.reduction_method == 'pca': reduced_updates_full = reduce_dimension_pca(flat_updates_matrix_full, self.pca_variance_ratio)
            elif self.reduction_method == 'dct': reduced_updates_full = reduce_dimension_dct(flat_updates_matrix_full, self.dct_components)
            else: reduced_updates_full = flat_updates_matrix_full
            if reduced_updates_full is None or reduced_updates_full.shape[0] != num_received:
                 aggregated_params_cpu_return = super().aggregate(); self.round_counter += 1; return aggregated_params_cpu_return

            current_diagram = self.topological_analyser.compute_diagram(reduced_updates_full)
            if current_diagram is None or current_diagram.shape[0] == 0:
                 self.prev_diagram = None; aggregated_params_cpu_return = super().aggregate(); self.round_counter += 1; return aggregated_params_cpu_return

            h0_diagram = current_diagram[current_diagram[:, 2] == 0]

            bottleneck_dist = np.nan
            if self.prev_diagram is not None and self.prev_diagram.shape[0] > 0 and h0_diagram.shape[0] > 0:
                finite_prev_h0 = self.prev_diagram[np.isfinite(self.prev_diagram[:, 1])]
                finite_curr_h0 = h0_diagram[np.isfinite(h0_diagram[:, 1])]
                if finite_prev_h0.shape[0] > 0 and finite_curr_h0.shape[0] > 0:
                     try:
                         bottleneck_dist = persim.bottleneck(finite_prev_h0[:, :2], finite_curr_h0[:, :2])
                         self.current_bottleneck_dist = bottleneck_dist # Store for logging
                         print(f" Bottleneck distance (Full Update): {bottleneck_dist:.4f}")
                     except Exception as persim_err:
                         print(f"Warning: persim.bottleneck failed: {persim_err}"); bottleneck_dist = np.nan

            if not np.isnan(bottleneck_dist) and bottleneck_dist > current_bottleneck_threshold:
                # perform_filtering = True # We note it, but don't act
                print(f" *** TDA Change Detected (Dist {bottleneck_dist:.4f} > Threshold {current_bottleneck_threshold:.4f}). (Observation Mode) ***")
            else:
                print(" No significant TDA change detected. Updating prev_diagram.")
                
            self.prev_diagram = h0_diagram
            # --- MODIFICATION: ALWAYS calculate bias stats, NEVER filter ---
            bias_vectors = {}
            valid_client_ids_bias = []
            for i, client_id in enumerate(client_ids_received):
                client_params = original_received_params[i]
                bias_vector = self._extract_bias_vector(client_params)
                if bias_vector is not None:
                    bias_vectors[client_id] = bias_vector
                    valid_client_ids_bias.append(client_id)
                else:
                    print(f" Warning: Could not extract bias vector for client ID {client_id}.")

            dists_from_median_map = {}
            valid_dists_this_round = []
            if len(valid_client_ids_bias) >= self.min_clients_for_defense:
                bias_matrix = np.vstack([bias_vectors[cid] for cid in valid_client_ids_bias])
                try:
                    median_bias_vector = np.median(bias_matrix, axis=0)
                    dists_from_median_valid = np.zeros(len(valid_client_ids_bias))
                    if self.bias_metric == 'cosine':
                         median_norm = np.linalg.norm(median_bias_vector)
                         for j, cid in enumerate(valid_client_ids_bias):
                             bias_v = bias_vectors[cid]; bias_v_norm = np.linalg.norm(bias_v)
                             if bias_v_norm < 1e-9 or median_norm < 1e-9: dists_from_median_valid[j] = 1.0
                             else: dists_from_median_valid[j] = cosine(bias_v, median_bias_vector)
                    else: # Euclidean
                         dists_from_median_valid = np.linalg.norm(bias_matrix - median_bias_vector, axis=1)

                    dists_from_median_map = {cid: dist for cid, dist in zip(valid_client_ids_bias, dists_from_median_valid)}
                    valid_dists_this_round = dists_from_median_valid

                    if len(valid_dists_this_round) > 0:
                         # Store stats for logging
                         self.current_bias_dist_stats['mean'] = np.mean(valid_dists_this_round)
                         self.current_bias_dist_stats['min'] = np.min(valid_dists_this_round)
                         self.current_bias_dist_stats['max'] = np.max(valid_dists_this_round)
                         self.current_bias_dist_stats['std'] = np.std(valid_dists_this_round)

                         print(f" Current Round DistFromMedian Bias Stats: Mean={self.current_bias_dist_stats['mean']:.4f}, Min={self.current_bias_dist_stats['min']:.4f}, Max={self.current_bias_dist_stats['max']:.4f}, Std={self.current_bias_dist_stats['std']:.4f}")
                         print("  Individual Bias Distances from Median:")
                         for cid in valid_client_ids_bias:
                             is_mal_str = " (Malicious)" if cid in self.malicious_ids_known else ""
                             print(f"    Client {cid}{is_mal_str}: {dists_from_median_map[cid]:.4f}")
                    
                except Exception as dist_err:
                    print(f" Error calculating bias distances from median: {dist_err}.")
            
            # --- REMOVED: All filtering logic ---
            # --- END MODIFICATION ---

        except Exception as e:
            print(f"!!! Error during analysis steps: {e}. Falling back to FedAvg. !!!")
            warnings.warn(f"Analysis Error: {e}", RuntimeWarning)
            self.prev_diagram = None

        # Aggregate ALL Clients
        print(f" Aggregating all {num_received} received clients (Observation Mode).")
        aggregated_params_cpu_return = super().aggregate() # This uses the *original* self.received_updates

        print(f"--- TopoBias Analysis Finished ---")
        self.round_counter += 1
        return aggregated_params_cpu_return

    # --- ADDED: close() method to close logger if it exists ---
    def close(self):
         """ Close the defense logger file if it exists. """
         if hasattr(self, 'defense_logger') and self.defense_logger:
             self.defense_logger.close()
             print("Closed defense logger.")
    # --- END ADDED ---

