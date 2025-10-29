import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import copy
import warnings
from torch.utils.data import DataLoader # Added DataLoader

# Import necessary components
from ..fl.server import FedAvgAggregator

# REMOVED: MAD function (observation only)

class AnalysisServer(FedAvgAggregator):
    """
    Analyzes client models based on behavior using a fixed batch from the testloader.
    Calculates average entropy and/or average Jacobian spectral norm over the batch.
    Performs NO filtering. Aggregates ALL received updates using standard FedAvg.
    """
    def __init__(self, model: torch.nn.Module, testloader: Optional[DataLoader]=None, device: Optional[torch.device]=None, defense_config: Optional[Dict] = None, **kwargs):
        super().__init__(model, testloader, device)
        if defense_config is None: defense_config = {}
        if testloader is None:
            raise ValueError("AnalysisServer requires a testloader to fetch probe batch.")

        # --- Analysis Selection ---
        self.analyze_entropy = defense_config.get('analyze_entropy', True)
        self.analyze_jacobian = defense_config.get('analyze_jacobian', True)
        self.probe_batch_size = defense_config.get('probe_batch_size', 64) # How many samples for analysis

        # --- Fetch and store one batch from testloader ---
        self.probe_batch_input: Optional[torch.Tensor] = None
        self.probe_batch_labels: Optional[torch.Tensor] = None
        try:
            # Create a dataloader with the specified batch size just for fetching
            probe_loader = DataLoader(testloader.dataset, batch_size=self.probe_batch_size, shuffle=True) # Shuffle to get a random batch
            self.probe_batch_input, self.probe_batch_labels = next(iter(probe_loader))
            # Move the batch to the server's device
            self.probe_batch_input = self.probe_batch_input.to(self.device)
            # Labels might not be needed but keep them just in case
            self.probe_batch_labels = self.probe_batch_labels.to(self.device)
            print(f"--- Initializing AnalysisServer (Test Batch Probing) ---")
            print(f" Fetched probe batch of size: {self.probe_batch_input.shape}")
            print(f" Analyze Entropy: {self.analyze_entropy}, Analyze Jacobian: {self.analyze_jacobian}")
            print(f"-------------------------------------------------------")

        except Exception as e:
            raise RuntimeError(f"Failed to fetch probe batch from testloader: {e}")

        # REMOVED: PGD parameters, base_random_input, probe_input_type

        self.round_counter = 0

    # REMOVED: _generate_probe_input method

    def _calculate_batch_entropy(self, model_instance: torch.nn.Module) -> Optional[float]:
        """ Calculates AVERAGE prediction entropy over the stored probe_batch_input. """
        if self.probe_batch_input is None: return None
        try:
            model_instance.eval()
            with torch.no_grad():
                logits = model_instance(self.probe_batch_input)
                probabilities = F.softmax(logits, dim=-1)
                # Calculate entropy per sample: - sum(p * log(p)) over classes
                entropy_per_sample = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=-1)
                # Return the average entropy over the batch
                avg_entropy = torch.mean(entropy_per_sample)
            return avg_entropy.item()
        except Exception as e:
            print(f"Error during batch entropy calculation: {e}")
            return None

    def _calculate_batch_jacobian_spectral_norm(self, model_instance: torch.nn.Module) -> Optional[float]:
        """ Calculates AVERAGE spectral norm of the Jacobian matrix over the probe_batch_input. """
        if self.probe_batch_input is None: return None

        model_instance.eval()
        # Ensure probe input requires grad for Jacobian calculation
        probe_input = self.probe_batch_input.clone().detach().requires_grad_(True)

        def model_wrapper(inp):
             # Handles batch input directly
             return model_instance(inp)

        try:
            # Compute Jacobian. Output shape: (Batch, NumClasses, Batch, C, H, W) or (B, N_Out, B, N_In)
            # Need to compute per-sample norms
            # Using torch.vmap might be efficient, but requires PyTorch 1.8+
            # Let's do a loop for broader compatibility, calculating norm for each sample in batch

            spectral_norms = []
            for i in range(probe_input.shape[0]): # Iterate over batch dimension
                single_input = probe_input[i:i+1] # Keep batch dim: (1, C, H, W)
                single_input = single_input.requires_grad_(True) # Ensure it requires grad

                # Jacobian for a single input
                jacobian_matrix_single = torch.autograd.functional.jacobian(model_wrapper, single_input, create_graph=False, strict=False)
                # Output shape: (1, NumClasses, 1, C, H, W) -> need to reshape to (NumClasses, C*H*W)
                try:
                    num_outputs = jacobian_matrix_single.shape[1]
                    num_inputs = single_input.numel()
                    # Reshape carefully - squeeze might remove batch dims needed
                    jacobian_2d_single = jacobian_matrix_single.permute(1, 0, *range(2, jacobian_matrix_single.dim())).reshape(num_outputs, num_inputs)
                except Exception as reshape_err:
                     print(f" Jacobian reshape error for sample {i} ({reshape_err}). Shape was: {jacobian_matrix_single.shape}. Input: {single_input.shape}. Skipping sample.")
                     continue # Skip this sample

                singular_values = torch.linalg.svdvals(jacobian_2d_single)
                spectral_norm = torch.max(singular_values)
                spectral_norms.append(spectral_norm.item())

            if not spectral_norms: # If loop failed for all samples
                return None

            # Return the average spectral norm over the batch
            avg_spectral_norm = np.mean(spectral_norms)
            return avg_spectral_norm

        except Exception as e:
            print(f"Error during batch Jacobian spectral norm calculation: {e}")
            return None


    def aggregate(self) -> Dict[str, torch.Tensor]:
        num_received = len(self.received_params)
        self.round_counter += 1
        print(f"\n--- AnalysisServer Aggregation (Round {self.round_counter}, Received: {num_received}) ---")

        original_received_params = self.received_params
        original_received_lens = self.received_lens

        analysis_results = {} # Store results per client index: {'avg_entropy': E, 'avg_jacobian_norm': J}

        print(" Analyzing received updates using test batch...")
        temp_model = copy.deepcopy(self.model).to(self.device)
        for i, client_params in enumerate(original_received_params):
            client_results = {'avg_entropy': np.nan, 'avg_jacobian_norm': np.nan} # Default NaN
            try:
                # Load state dict logic (unchanged)
                current_model_keys = set(temp_model.state_dict().keys())
                client_keys = set(client_params.keys())
                strict_load = current_model_keys == client_keys
                temp_model.load_state_dict(client_params, strict=strict_load)
                if not strict_load:
                    print(f" Warning: State dict keys mismatch for update {i}. Loaded with strict=False.")

                # REMOVED: Call to _generate_probe_input

                if self.analyze_entropy:
                    avg_entropy = self._calculate_batch_entropy(temp_model) # Now calculates avg over batch
                    if avg_entropy is not None: client_results['avg_entropy'] = avg_entropy

                if self.analyze_jacobian:
                    avg_jac_norm = self._calculate_batch_jacobian_spectral_norm(temp_model) # Now calculates avg over batch
                    if avg_jac_norm is not None: client_results['avg_jacobian_norm'] = avg_jac_norm

            except Exception as load_err:
                 print(f" Warning: Failed analysis for update {i}: {load_err}.")
            finally:
                 analysis_results[i] = client_results


        # --- Print Analysis Results ---
        print(" Analysis Results per Client (Index) - Averaged over test batch:")
        for idx, results in analysis_results.items():
             entropy_str = f"{results['avg_entropy']:.4f}" if not np.isnan(results['avg_entropy']) else "N/A"
             jac_norm_str = f"{results['avg_jacobian_norm']:.4f}" if not np.isnan(results['avg_jacobian_norm']) else "N/A"
             print(f"  Client Index {idx}: AvgEntropy={entropy_str}, AvgJacobianNorm={jac_norm_str}")

        if analysis_results:
             if self.analyze_entropy:
                  all_entropies = np.array([r['avg_entropy'] for r in analysis_results.values() if not np.isnan(r['avg_entropy'])])
                  if len(all_entropies) > 0:
                       print(f" Avg Entropy Stats: Min={np.min(all_entropies):.4f}, Mean={np.mean(all_entropies):.4f}, Max={np.max(all_entropies):.4f}, StdDev={np.std(all_entropies):.4f}")
             if self.analyze_jacobian:
                  all_jac_norms = np.array([r['avg_jacobian_norm'] for r in analysis_results.values() if not np.isnan(r['avg_jacobian_norm'])])
                  if len(all_jac_norms) > 0:
                       print(f" Avg Jacobian Norm Stats: Min={np.min(all_jac_norms):.4f}, Mean={np.mean(all_jac_norms):.4f}, Max={np.max(all_jac_norms):.4f}, StdDev={np.std(all_jac_norms):.4f}")

        # --- Aggregate ALL Clients using super().aggregate() ---
        print(f" Aggregating all {num_received} received clients using standard FedAvg.")
        self.received_params = original_received_params
        self.received_lens = original_received_lens
        aggregated_params_cpu_return = super().aggregate()

        print(f"--- AnalysisServer Aggregation Finished ---")
        return aggregated_params_cpu_return

