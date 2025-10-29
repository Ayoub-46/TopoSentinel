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

try:
    import persim
except ImportError:
    print("Error: persim library not found. Please install using 'pip install persim'")
    persim = None

class TopologicalBiasDefenseServer(FedAvgAggregator):
    """
    Hybrid Defense:
    - Inter-round: TDA H0 bottleneck distance (Decaying Threshold trigger).
    - Intra-round (if triggered): Filters bias vectors based on distance from median,
      using an adaptively learned [min, max] interval based on benign history percentiles.
    """
    def __init__(self, model: torch.nn.Module, testloader=None, device: Optional[torch.device]=None, defense_config: Optional[Dict] = None, **kwargs):
        # Pass kwargs to parent (e.g., for logger)
        super().__init__(model, testloader, device, **kwargs) 
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

        # Bias Filtering Parameters (Adaptive Interval)
        self.bias_metric = defense_config.get('bias_metric', 'cosine').lower()
        self.bias_history_window = defense_config.get('bias_history_window', 20) # Rounds
        self.bias_interval_lower_percentile = defense_config.get('bias_interval_lower_percentile', 5.0)
        self.bias_interval_upper_percentile = defense_config.get('bias_interval_upper_percentile', 95.0)
        self.bias_interval_margin = defense_config.get('bias_interval_margin', 0.01) # Small margin
        self.bias_fallback_interval = defense_config.get('bias_fallback_interval', [0.0, 0.5]) # Fallback
        self.min_clients_for_defense = defense_config.get('min_clients_for_defense', 3)
        self.min_bias_history_size = max(defense_config.get('min_bias_history_size', 50), self.min_clients_for_defense * 3) # Min data points

        print(f"--- Initializing Hybrid TDA-Bias Defense Server (Adaptive Interval Filter) ---")
        print(f" TDA Trigger: Reduction={self.reduction_method}, Metric=magnitude_cosine(alpha={self.tda_metric_alpha})")
        print(f"  Decaying Bottleneck: Initial={self.bottleneck_initial_threshold}, DecayRate={self.bottleneck_decay_rate}, Min={self.bottleneck_min_threshold}")
        print(f" Bias Filtering: Metric=DistFromMedian({self.bias_metric})")
        print(f"  Adaptive Bias Interval: Percentiles=[{self.bias_interval_lower_percentile}, {self.bias_interval_upper_percentile}], Margin={self.bias_interval_margin}, MinHistSize={self.min_bias_history_size}, Fallback={self.bias_fallback_interval}")
        print(f" Min Clients: {self.min_clients_for_defense}")
        print(f"-----------------------------------------------------------------------------")

        self.topological_analyser = TopologicalAnalyser(
            homology_dimensions=(0,),
            metric=magnitude_cosine_distance,
            metric_params={'alpha': self.tda_metric_alpha}
        )
        self.prev_diagram: Optional[np.ndarray] = None
        self.round_counter = 0
        self.bias_distance_history = deque(maxlen=int(self.bias_history_window * self.min_clients_for_defense * 2.5))


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

    def aggregate(self) -> Dict[str, torch.Tensor]:
        num_received = len(self.received_updates)
        current_round_print = self.round_counter + 1
        print(f"\n--- Hybrid TDA-Bias Aggregation (Round {current_round_print}, Received: {num_received}) ---")

        client_ids_received = list(self.received_updates.keys())
        original_received_params = [self.received_updates[cid]['params'] for cid in client_ids_received]
        original_received_lens = [self.received_updates[cid]['length'] for cid in client_ids_received]
        
        client_ids_to_keep = copy.copy(client_ids_received) 
        perform_filtering = False

        if num_received < self.min_clients_for_defense:
            print(f" Updates < min clients. Falling back to FedAvg.")
            aggregated_params_cpu_return = super().aggregate() # Clears original self.received_updates
            self.prev_diagram = None; self.round_counter += 1
            return aggregated_params_cpu_return

        current_bottleneck_threshold = max(
            self.bottleneck_min_threshold,
            self.bottleneck_initial_threshold * (self.bottleneck_decay_rate ** self.round_counter)
        )
        print(f" Current Decaying Bottleneck Threshold: {current_bottleneck_threshold:.4f}")

        aggregated_params_cpu_return = {}

        try:
            # TDA Analysis on FULL Update Deltas (for Trigger)
            current_global_params = self.get_params()
            client_deltas_full = []; metadata = None
            for client_params in original_received_params: # Use list of params
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
                         print(f" Bottleneck distance (Full Update): {bottleneck_dist:.4f}")
                     except Exception as persim_err:
                         print(f"Warning: persim.bottleneck failed: {persim_err}"); bottleneck_dist = np.nan

            if not np.isnan(bottleneck_dist) and bottleneck_dist > current_bottleneck_threshold:
                perform_filtering = True
                print(f" *** TDA Change Detected. Triggering BIAS filtering. ***")
            else:
                print(" No significant TDA change detected. Updating prev_diagram.")
                self.prev_diagram = h0_diagram

            bias_vectors = {} # map client_id -> bias_vector
            valid_client_ids_bias = []
            for i, client_id in enumerate(client_ids_received):
                client_params = original_received_params[i] # Get params by original index
                bias_vector = self._extract_bias_vector(client_params)
                if bias_vector is not None:
                    bias_vectors[client_id] = bias_vector
                    valid_client_ids_bias.append(client_id)
                else:
                    print(f" Warning: Could not extract bias vector for client ID {client_id}.")

            dists_from_median_map = {} # map client_id -> dist_from_median
            valid_dists_this_round = [] # List of valid distances
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
                         print(f" Current Round DistFromMedian Bias Stats: Median={np.median(valid_dists_this_round):.4f}, Min={np.min(valid_dists_this_round):.4f}, Max={np.max(valid_dists_this_round):.4f}")

                except Exception as dist_err:
                    print(f" Error calculating bias distances from median: {dist_err}.")

            if perform_filtering:
                print(f"--- Filtering based on Learned Bias Interval ({self.bias_interval_lower_percentile}-{self.bias_interval_upper_percentile}th percentile) ---")
                current_benign_interval = self.bias_fallback_interval
                history_size = len(self.bias_distance_history)

                if history_size >= self.min_bias_history_size:
                     history_arr = np.array(list(self.bias_distance_history))
                     try:
                         lower_bound = np.percentile(history_arr, self.bias_interval_lower_percentile)
                         upper_bound = np.percentile(history_arr, self.bias_interval_upper_percentile)
                         lower_bound_margin = max(0.0, lower_bound - self.bias_interval_margin)
                         upper_bound_margin = upper_bound + self.bias_interval_margin
                         if upper_bound_margin <= lower_bound_margin: upper_bound_margin = lower_bound_margin + 1e-6
                         current_benign_interval = [lower_bound_margin, upper_bound_margin]
                         print(f" Learned Benign Interval (from {history_size} points, margin {self.bias_interval_margin}): [{current_benign_interval[0]:.4f}, {current_benign_interval[1]:.4f}]")
                     except IndexError:
                         print(f" Error calculating percentiles. Using fallback interval.")
                         current_benign_interval = self.bias_fallback_interval
                else:
                     print(f" Not enough bias history ({history_size} points). Using fallback interval: {self.bias_fallback_interval}")
                     current_benign_interval = self.bias_fallback_interval

                if dists_from_median_map: # Check if distances were calculated
                     outlier_client_ids = set()
                     for client_id in client_ids_received:
                         dist = dists_from_median_map.get(client_id, np.inf) # Get dist or treat as outlier
                         if not (current_benign_interval[0] <= dist <= current_benign_interval[1]):
                             outlier_client_ids.add(client_id)
                     
                     client_ids_to_keep = [cid for cid in client_ids_received if cid not in outlier_client_ids]
                     num_filtered = len(outlier_client_ids)
                     
                     if num_filtered > 0:
                         print(f" Identified {num_filtered} outliers via Bias Adaptive Interval (Client IDs: {sorted(list(outlier_client_ids))}). Keeping {len(client_ids_to_keep)}.");
                    
                     else:
                         print(" No outliers detected by bias vector Adaptive Interval.")
                         
                     if len(client_ids_to_keep) == 0:
                         print("Warning: All clients flagged by bias filter. Keeping all.");
                         client_ids_to_keep = client_ids_received
                else:
                    print(" Bias distances not calculated. Keeping all clients.")
                    client_ids_to_keep = client_ids_received
            else:
                 # Update History (If round is benign and distances were calculated)
                 if len(valid_dists_this_round) > 0:
                     self.bias_distance_history.extend(valid_dists_this_round)
                     print(f" Round deemed benign. Added {len(valid_dists_this_round)} bias distances to history (now {len(self.bias_distance_history)} points).")
                 # client_ids_to_keep remains all clients

        except Exception as e:
            print(f"!!! Error during defense steps: {e}. Falling back to FedAvg. !!!")
            warnings.warn(f"Defense Error: {e}", RuntimeWarning)
            self.prev_diagram = None
            client_ids_to_keep = client_ids_received # Fallback: keep everyone

        # Aggregate Kept Clients using super().aggregate()
        if len(client_ids_to_keep) < num_received: print(f" Aggregating {len(client_ids_to_keep)} clients after filtering.")
        else: print(f" Aggregating all {num_received} received clients.")
        
        updates_to_aggregate = {}
        for client_id in client_ids_to_keep:
             if client_id in self.received_updates:
                 updates_to_aggregate[client_id] = self.received_updates[client_id]
             
        if not updates_to_aggregate:
            print("Warning: No clients left after filtering. Skipping aggregation.")
            aggregated_params_cpu_return = self.get_params()
            self.received_updates = {}
        else:
            self.received_updates = updates_to_aggregate
            aggregated_params_cpu_return = super().aggregate() # Use parent logic, clears temp buffers

        print(f"--- Hybrid TDA-Bias Aggregation Finished ---")
        self.round_counter += 1
        return aggregated_params_cpu_return

