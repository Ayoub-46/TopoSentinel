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
from ..persistent_homology.metrics import magnitude_cosine_distance
from .metrics_mixin import DefenseMetricsMixin 

try:
    import persim
except ImportError:
    print("Error: persim library not found. Please install using 'pip install persim'")
    persim = None

def _calculate_update_l2_norms(updates: Dict, global_params: Dict, client_ids: List[int]) -> Dict[int, float]:
    """Calculates the L2 norm (Euclidean distance) of the full update delta (local - global) for each client."""
    distances = {}
    for cid in client_ids:
        local_params = updates[cid]['params']
        flat_delta = []
        for name in local_params:
            if name in global_params:
                # Calculate delta: local - global
                diff = local_params[name].cpu() - global_params[name].cpu()
                flat_delta.append(diff.flatten())
        
        if flat_delta:
            # L2 norm of the concatenated delta vector
            l2_norm = torch.linalg.norm(torch.cat(flat_delta)).item()
            distances[cid] = l2_norm
        else:
            distances[cid] = 0.0
    return distances

class TopologicalRobustDefenseServer(DefenseMetricsMixin, FedAvgAggregator):
    """
    Hybrid Defense (Bias-Only TDA):
    - Inter-round: TDA H0 bottleneck distance on BIAS DELTAS (Decaying Threshold trigger).
    - Intra-round (if triggered): Filters bias vectors based on distance from median,
      using an adaptively learned [min, max] interval based on benign history percentiles.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        
        defense_config = kwargs.get('defense_config', None)
        if defense_config is None and len(args) >= 4:
            defense_config = args[3]
            
        if defense_config is None: defense_config = {}
        if persim is None: raise ImportError("persim library is required but not installed.")

        # TDA Parameters 
        self.bias_metric = defense_config.get('bias_metric', 'euclidean').lower() 
        self.bottleneck_initial_threshold = defense_config.get('bottleneck_initial_threshold', 0.8)
        self.bottleneck_decay_rate = defense_config.get('bottleneck_decay_rate', 0.99)
        self.bottleneck_min_threshold = defense_config.get('bottleneck_min_threshold', 0.01)

        # Bias Filtering Parameters (Adaptive Interval)
        self.bias_history_window = defense_config.get('bias_history_window', 20) # Rounds
        self.bias_interval_lower_percentile = defense_config.get('bias_interval_lower_percentile', 5.0)
        self.bias_interval_upper_percentile = defense_config.get('bias_interval_upper_percentile', 95.0)
        self.bias_interval_margin = defense_config.get('bias_interval_margin', 0.01) # Small margin
        self.bias_fallback_interval = defense_config.get('bias_fallback_interval', [0.0, 0.5]) # Fallback
        self.min_clients_for_defense = defense_config.get('min_clients_for_defense', 3)
        self.min_bias_history_size = max(defense_config.get('min_bias_history_size', 50), self.min_clients_for_defense * 3) # Min data points

        print(f"--- Initializing Hybrid TDA-Bias Defense Server (Bias TDA) ---")
        print(f" TDA Trigger: Metric={self.bias_metric} (on Bias Deltas)")
        print(f"  Decaying Bottleneck: Initial={self.bottleneck_initial_threshold}, DecayRate={self.bottleneck_decay_rate}, Min={self.bottleneck_min_threshold}")
        print(f" Bias Filtering: Metric=DistFromMedian({self.bias_metric}) (on Bias Deltas)")
        print(f"  Adaptive Bias Interval: Percentiles=[{self.bias_interval_lower_percentile}, {self.bias_interval_upper_percentile}], Margin={self.bias_interval_margin}, MinHistSize={self.min_bias_history_size}, Fallback={self.bias_fallback_interval}")
        print(f" Min Clients: {self.min_clients_for_defense}")
        print(f"-----------------------------------------------------------------------------")
        
    
        # Select the metric function for TDA
        if self.bias_metric == 'magnitude_cosine':
            tda_metric_func = magnitude_cosine_distance
            tda_metric_params = {'alpha': 0.5} 
        elif self.bias_metric == 'cosine' or self.bias_metric == 'euclidean':
            tda_metric_func = self.bias_metric 
            tda_metric_params = {}
        else:
             print(f"Warning: Unknown bias_metric '{self.bias_metric}' for TDA. Defaulting to 'euclidean'.")
             tda_metric_func = 'euclidean'
             tda_metric_params = {}

        self.topological_analyser = TopologicalAnalyser(
            homology_dimensions=(0,),
            metric=tda_metric_func,
            metric_params=tda_metric_params
        )
        
        self.prev_diagram: Optional[np.ndarray] = None
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
        client_ids_received_set = set(client_ids_received) # Define set early for metrics
        original_received_params = [self.received_updates[cid]['params'] for cid in client_ids_received]
        original_received_lens = [self.received_updates[cid]['length'] for cid in client_ids_received]
        
        client_ids_to_keep = copy.copy(client_ids_received) 
        perform_filtering = False

        if num_received < self.min_clients_for_defense:
            print(f" Updates < min clients. Falling back to FedAvg.")
            
            self.update_defense_metrics(client_ids_received_set, rejected_client_ids=set())
            
            aggregated_params_cpu_return = super().aggregate() 
            self.prev_diagram = None; self.round_counter += 1
            return aggregated_params_cpu_return

        current_bottleneck_threshold = max(
            self.bottleneck_min_threshold,
            self.bottleneck_initial_threshold * (self.bottleneck_decay_rate ** self.round_counter)
        )
        print(f" Current Decaying Bottleneck Threshold: {current_bottleneck_threshold:.4f}")

        aggregated_params_cpu_return = {}

        try:
            # 1. Extract Bias Vectors and Deltas
            current_global_params = self.get_params()
            global_bias_vector = self._extract_bias_vector(current_global_params)
            
            client_bias_vectors = {} # map client_id -> bias_vector
            valid_client_ids_bias = []

            if global_bias_vector is None:
                print("Warning: Could not extract global bias vector. Skipping defense.")
                raise Exception("Failed to extract global bias vector.")

            for i, client_id in enumerate(client_ids_received):
                client_params = original_received_params[i]
                bias_vector = self._extract_bias_vector(client_params)
                if bias_vector is not None:
                    client_bias_vectors[client_id] = bias_vector
                    valid_client_ids_bias.append(client_id)
                else:
                    print(f" Warning: Could not extract bias vector for client ID {client_id}.")

            if len(valid_client_ids_bias) < self.min_clients_for_defense:
                 print("Warning: Not enough valid bias vectors. Skipping defense.")
                 raise Exception("Not enough valid bias vectors for defense.")

            # Create bias *delta* matrix (client_vector - global_vector) for TDA
            # Note: We only analyze clients where bias extraction succeeded
            bias_delta_matrix = np.vstack([client_bias_vectors[cid] for cid in valid_client_ids_bias]) - global_bias_vector
            print(f" Extracted bias delta matrix shape: {bias_delta_matrix.shape}")

            # 2. Compute H0 Diagram (on Bias Deltas)
            current_diagram = self.topological_analyser.compute_diagram(bias_delta_matrix)
            if current_diagram is None or current_diagram.shape[0] == 0:
                 print("Warning: Failed to compute persistence diagram on bias deltas. Skipping TDA.")
                 self.prev_diagram = None
                 # Still proceed to aggregation, but filtering remains False
                 pass # Continue to step 4 in the main 'try' block below

            else:
                 h0_diagram = current_diagram[current_diagram[:, 2] == 0]

                 # 3. Inter-round Analysis (Bottleneck Distance)
                 bottleneck_dist = np.nan
                 if self.prev_diagram is not None and self.prev_diagram.shape[0] > 0 and h0_diagram.shape[0] > 0:
                     finite_prev_h0 = self.prev_diagram[np.isfinite(self.prev_diagram[:, 1])]
                     finite_curr_h0 = h0_diagram[np.isfinite(h0_diagram[:, 1])]
                     if finite_prev_h0.shape[0] > 0 and finite_curr_h0.shape[0] > 0:
                          try:
                              bottleneck_dist = persim.bottleneck(finite_prev_h0[:, :2], finite_curr_h0[:, :2])
                              print(f" Bottleneck distance (Bias Deltas): {bottleneck_dist:.4f}")
                          except Exception as persim_err:
                              print(f"Warning: persim.bottleneck failed: {persim_err}"); bottleneck_dist = np.nan

                 # 4. Trigger Filtering
                 if not np.isnan(bottleneck_dist) and bottleneck_dist > current_bottleneck_threshold:
                     perform_filtering = True
                     print(f" *** TDA Change Detected (Bias Deltas). Triggering BIAS filtering. ***")
                 else:
                     print(" No significant TDA change detected. Updating prev_diagram.")
                     self.prev_diagram = h0_diagram

            # Intra-round Filtering
            
            dists_from_median_map = {} # map client_id -> dist_from_median
            valid_dists_this_round = [] # List of valid distances
            
            # bias_matrix is on valid_client_ids_bias
            bias_matrix_abs = np.vstack([client_bias_vectors[cid] for cid in valid_client_ids_bias])
            try:
                median_bias_vector = np.median(bias_matrix_abs, axis=0)
                dists_from_median_valid = np.zeros(len(valid_client_ids_bias))
                if self.bias_metric == 'cosine':
                     median_norm = np.linalg.norm(median_bias_vector)
                     for j, cid in enumerate(valid_client_ids_bias):
                         bias_v = client_bias_vectors[cid]; bias_v_norm = np.linalg.norm(bias_v)
                         if bias_v_norm < 1e-9 or median_norm < 1e-9: dists_from_median_valid[j] = 1.0
                         else: dists_from_median_valid[j] = cosine(bias_v, median_bias_vector)
                else: # Euclidean
                     dists_from_median_valid = np.linalg.norm(bias_matrix_abs - median_bias_vector, axis=1)

                dists_from_median_map = {cid: dist for cid, dist in zip(valid_client_ids_bias, dists_from_median_valid)}
                valid_dists_this_round = dists_from_median_valid

                if len(valid_dists_this_round) > 0:
                     print(f" Current Round DistFromMedian Bias *Vector* Stats: Median={np.median(valid_dists_this_round):.4f}, Min={np.min(valid_dists_this_round):.4f}, Max={np.max(valid_dists_this_round):.4f}")

            except Exception as dist_err:
                print(f" Error calculating bias distances from median: {dist_err}.")
                dists_from_median_map = {} 

            # Filtering Logic 
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

                if dists_from_median_map: 
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
                 if len(valid_dists_this_round) > 0:
                     self.bias_distance_history.extend(valid_dists_this_round)
                     print(f" Round deemed benign. Added {len(valid_dists_this_round)} bias distances to history (now {len(self.bias_distance_history)} points).")

        except Exception as e:
            print(f"!!! Error during defense steps: {e}. Falling back to FedAvg. !!!")
            warnings.warn(f"Defense Error: {e}", RuntimeWarning)
            self.prev_diagram = None
            client_ids_to_keep = client_ids_received # Fallback: keep everyone

        client_ids_to_keep_set = set(client_ids_to_keep)
        rejected_client_ids = client_ids_received_set - client_ids_to_keep_set
        
        self.update_defense_metrics(
            client_ids_received=client_ids_received_set,
            rejected_client_ids=rejected_client_ids
        )

        client_update_l2_norms = _calculate_update_l2_norms(self.received_updates, current_global_params, client_ids_to_keep)
        
        kept_norms = [client_update_l2_norms[cid] for cid in client_ids_to_keep if client_update_l2_norms[cid] > 0]
        
        if not kept_norms:
             print("Warning: All kept clients had zero update norms. Skipping aggregation.")
             self.received_updates = {}
             self.round_counter += 1
             return self.get_params()

        clip_norm = torch.median(torch.tensor(kept_norms)).item()
        print(f" Adaptive Clipping Norm (Median L2 Delta of Kept Clients): {clip_norm:.4f}")

        aggregated_delta = {name: torch.zeros_like(param).to('cpu') for name, param in current_global_params.items()}
        total_kept_samples = sum(self.received_updates[cid]['length'] for cid in client_ids_to_keep)
        
        if total_kept_samples == 0:
            self.received_updates = {}
            self.round_counter += 1
            return self.get_params()

        for cid in client_ids_to_keep:
            local_params = self.received_updates[cid]['params'] 
            num_samples = self.received_updates[cid]['length']
            weight = num_samples / total_kept_samples
            
            delta = {name: local_params[name] - current_global_params[name] for name in local_params if name in current_global_params}
            
            client_dist = client_update_l2_norms.get(cid, 0.0)
            if client_dist > clip_norm:
                scaling_factor = clip_norm / (client_dist + 1e-10) 
                for name in delta:
                    if not name.endswith('num_batches_tracked'):
                        delta[name].mul_(scaling_factor)

            for name, param_delta in delta.items():
                if name in aggregated_delta:
                    aggregated_delta[name].add_(param_delta, alpha=weight)

        new_global_state = self.model.state_dict()
        for name, param in new_global_state.items():
            if name in aggregated_delta:
                param.add_(aggregated_delta[name].to(self.device))
        
        self.set_params(new_global_state) 
        self.received_updates = {} 

        print(f"--- Topological Robust Aggregation Finished ---")
        self.round_counter += 1
        return self.get_params()


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