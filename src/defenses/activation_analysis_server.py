import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import copy
import warnings
from torch.utils.data import DataLoader
from scipy.spatial.distance import pdist, squareform, cosine # Keep for analysis stats

# Import necessary components
from ..fl.server import FedAvgAggregator

# REMOVED: MAD function (observation only)
# REMOVED: get_layer function (not needed for bias analysis)

class BiasAnalysisServer(FedAvgAggregator):
    """
    Analyzes client models by comparing their concatenated BIAS parameters.
    Calculates stats on bias vector norms and distances.
    Performs NO filtering. Aggregates ALL received updates using standard FedAvg.
    """
    def __init__(self, model: torch.nn.Module, testloader: Optional[DataLoader]=None, device: Optional[torch.device]=None, defense_config: Optional[Dict] = None, **kwargs):
        super().__init__(model, testloader, device) # Pass testloader even if not used directly now
        if defense_config is None: defense_config = {}

        # --- Analysis Parameters ---
        # Choose distance metric for comparing bias vectors ('cosine' or 'euclidean')
        self.bias_metric = defense_config.get('bias_metric', 'cosine').lower()

        print(f"--- Initializing BiasAnalysisServer ---") # Updated Name
        print(f" Bias comparison metric: {self.bias_metric}")
        print(f"---------------------------------------")

        self.round_counter = 0

    def _extract_bias_vector(self, model_instance: nn.Module) -> Optional[np.ndarray]:
        """
        Extracts all bias parameters from a model and concatenates them into a flat NumPy array.
        """
        bias_tensors = []
        try:
            model_instance.eval() # Ensure consistent state (e.g., for BN)
            with torch.no_grad():
                for name, param in model_instance.named_parameters():
                    if 'bias' in name: # Simple check for bias parameter names
                        bias_tensors.append(param.detach().cpu().flatten())

            if not bias_tensors:
                print("Warning: No bias parameters found in the model.")
                return None

            # Concatenate all found bias tensors and convert to NumPy
            flat_bias_vector = torch.cat(bias_tensors).numpy().astype(np.float64)
            return flat_bias_vector

        except Exception as e:
            print(f"Error extracting bias vector: {e}")
            return None


    def aggregate(self) -> Dict[str, torch.Tensor]:
        num_received = len(self.received_params)
        self.round_counter += 1
        print(f"\n--- BiasAnalysis Aggregation (Round {self.round_counter}, Received: {num_received}) ---") # Updated Name

        original_received_params = self.received_params
        original_received_lens = self.received_lens

        bias_vectors = {} # index -> bias_vector (np.ndarray)

        # --- ADDED: Extract bias vector from the global model (reference) ---
        print(" Extracting bias vector from current global model (as reference)...")
        # self.model holds the state from the end of the previous round
        global_bias_vector_ref = self._extract_bias_vector(self.model)
        if global_bias_vector_ref is None:
            print(" Warning: Could not extract global bias vector. Will skip 'Dist from Global' analysis.")
        # --- END ADDED ---

        print(f" Analyzing bias vectors for received updates...") # Updated print
        temp_model = copy.deepcopy(self.model).to(self.device)
        for i, client_params in enumerate(original_received_params):
            try:
                # Load state dict
                current_model_keys = set(temp_model.state_dict().keys()); client_keys = set(client_params.keys())
                strict_load = current_model_keys == client_keys; temp_model.load_state_dict(client_params, strict=strict_load)
                if not strict_load: print(f" Warning: State dict keys mismatch for update {i}.")

                # --- Extract Bias Vector ---
                bias_vector = self._extract_bias_vector(temp_model)
                if bias_vector is not None:
                    bias_vectors[i] = bias_vector
                else:
                    print(f" Warning: Could not extract bias vector for client index {i}.")

            except Exception as load_err:
                 print(f" Warning: Failed analysis for update {i}: {load_err}.")


        # --- Print Analysis Stats ---
        if len(bias_vectors) > 1:
            print(" Bias Vector Analysis Stats:")
            bias_matrix = np.vstack(list(bias_vectors.values())) # (n_clients, bias_dim)
            print(f"  Extracted bias vectors shape: {bias_matrix.shape}")

            # --- ADDED: L2 Norm Analysis ---
            l2_norms = np.linalg.norm(bias_matrix, ord=2, axis=1)
            print(f"  L2 Norm of Bias Vectors:")
            print(f"  Norms: {l2_norms}")
            print(f"   Min={np.min(l2_norms):.4f}, Mean={np.mean(l2_norms):.4f}, Max={np.max(l2_norms):.4f}, StdDev={np.std(l2_norms):.4f}")
            # --- END ADDED ---

            # Calculate pairwise distances
            try:
                pairwise_distances = pdist(bias_matrix, metric=self.bias_metric)
                dist_matrix = squareform(pairwise_distances)
                np.fill_diagonal(dist_matrix, np.inf)

                # Nearest Neighbor Distances
                nn_distances = np.min(dist_matrix, axis=1)
                print(f"  Nearest Neighbor Bias Vector Dist ({self.bias_metric}):")
                print(f"  Distances: {nn_distances}")
                print(f"   Min={np.min(nn_distances):.4f}, Mean={np.mean(nn_distances):.4f}, Max={np.max(nn_distances):.4f}, StdDev={np.std(nn_distances):.4f}")

                # Distances from Median Bias Vector
                median_bias_vector = np.median(bias_matrix, axis=0)
                if self.bias_metric == 'cosine':
                     dists_from_median = np.array([cosine(bias_v, median_bias_vector) if (np.linalg.norm(bias_v)>1e-9 and np.linalg.norm(median_bias_vector)>1e-9) else 1.0 for bias_v in bias_matrix])
                else: # Assume Euclidean
                     dists_from_median = np.linalg.norm(bias_matrix - median_bias_vector, axis=1)
                print(f"  Distance from Median Bias Vector ({self.bias_metric}):")
                print(f"  Distances: {dists_from_median}")
                print(f"   Min={np.min(dists_from_median):.4f}, Mean={np.mean(dists_from_median):.4f}, Max={np.max(dists_from_median):.4f}, StdDev={np.std(dists_from_median):.4f}")

                # --- ADDED: Distance from Global Ref Analysis ---
                if global_bias_vector_ref is not None:
                    if self.bias_metric == 'cosine':
                        global_ref_norm = np.linalg.norm(global_bias_vector_ref)
                        dists_from_global = np.array([cosine(bias_v, global_bias_vector_ref) if (np.linalg.norm(bias_v)>1e-9 and global_ref_norm>1e-9) else 1.0 for bias_v in bias_matrix])
                    else: # Assume Euclidean
                        dists_from_global = np.linalg.norm(bias_matrix - global_bias_vector_ref, axis=1)

                    print(f"  Distance from *Global Ref* Bias Vector ({self.bias_metric}):")
                    print(f"  Distances: {dists_from_global}")
                    print(f"   Min={np.min(dists_from_global):.4f}, Mean={np.mean(dists_from_global):.4f}, Max={np.max(dists_from_global):.4f}, StdDev={np.std(dists_from_global):.4f}")
                else:
                    print("  Skipping 'Distance from Global Ref' analysis (ref vector not extracted).")
                # --- END ADDED ---

            except Exception as dist_err:
                 print(f" Error calculating bias distance stats: {dist_err}")

        elif len(bias_vectors) == 1:
            print(" Only one valid bias vector extracted. Cannot compute distance stats.")
        else:
            print(" No valid bias vectors extracted.")


        # --- Aggregate ALL Clients ---
        print(f" Aggregating all {num_received} received clients using standard FedAvg.")
        self.received_params = original_received_params
        self.received_lens = original_received_lens
        aggregated_params_cpu_return = super().aggregate()

        print(f"--- BiasAnalysis Aggregation Finished ---") # Updated Name
        return aggregated_params_cpu_return
