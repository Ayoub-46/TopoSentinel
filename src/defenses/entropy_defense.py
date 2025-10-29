import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import copy
import warnings

# Import necessary components
from ..fl.server import FedAvgAggregator

# REMOVED: MAD function (observation only)

class EntropyDefenseServer(FedAvgAggregator):
    """
    Observes model behavior by calculating prediction entropy on inputs
    optimized via PGD to MINIMIZE entropy (maximize confidence).
    Performs NO filtering. Aggregates ALL received updates using standard FedAvg.
    """
    def __init__(self, model: torch.nn.Module, testloader=None, device: Optional[torch.device]=None, defense_config: Optional[Dict] = None, **kwargs):
        super().__init__(model, testloader, device)
        if defense_config is None: defense_config = {}

        # --- Input Shape ---
        input_shape_list = defense_config.get('input_shape')
        if input_shape_list is None or not isinstance(input_shape_list, list):
            raise ValueError("EntropyDefense requires 'input_shape' (e.g., [1, 3, 32, 32]) specified.")
        if input_shape_list[0] != 1: input_shape_list[0] = 1 # Ensure batch size 1
        self.random_input_shape = tuple(input_shape_list)

        # --- Base random input (starting point for optimization) ---
        self.base_random_input = torch.randn(self.random_input_shape)

        # --- PGD Optimization Parameters ---
        self.pgd_epsilon = defense_config.get('pgd_epsilon', 0.1)  # Allow larger search space maybe?
        self.pgd_steps = defense_config.get('pgd_steps', 40)      # More steps might be needed
        self.pgd_step_size = defense_config.get('pgd_step_size', 0.01) # Step size
        self.input_clip_min = defense_config.get('input_clip_min', 0.0)
        self.input_clip_max = defense_config.get('input_clip_max', 1.0)

        print(f"--- Initializing EntropyObserverServer (Confident Input Probing) ---") # Updated Name
        print(f" Input Shape (from config): {self.random_input_shape}")
        print(f" PGD Params (for Min Entropy): Epsilon={self.pgd_epsilon}, Steps={self.pgd_steps}, StepSize={self.pgd_step_size}")
        print(f"-----------------------------------------------------------------")

        self.round_counter = 0

    def _generate_confident_input(self, model_instance: torch.nn.Module) -> torch.Tensor:
        """
        Generates an input likely to yield a confident prediction (low entropy)
        by perturbing a base random input using PGD targeted to MINIMIZE entropy
        (by maximizing the highest logit).
        """
        model_instance.eval()
        x = self.base_random_input.clone().detach().to(self.device).requires_grad_(True)
        x_orig = x.detach()

        for _ in range(self.pgd_steps):
            if x.grad is not None:
                x.grad.detach_()
                x.grad.zero_()

            logits = model_instance(x)
            # --- Loss = Negative of the maximum logit ---
            # Maximizing the max logit pushes probability towards 1 for one class, minimizing entropy.
            loss = -torch.max(logits, dim=-1).values

            loss.backward()

            # --- PGD Step: Move in the NEGATIVE gradient direction to MINIMIZE loss ---
            with torch.no_grad(): # Ensure update steps aren't tracked
                 x_adv = x - self.pgd_step_size * x.grad.sign() # SUBTRACT gradient sign

                 # Projection step 1: Clip to epsilon ball
                 eta = torch.clamp(x_adv - x_orig, min=-self.pgd_epsilon, max=self.pgd_epsilon)
                 # Projection step 2: Clip to valid input range
                 x_new = torch.clamp(x_orig + eta, min=self.input_clip_min, max=self.input_clip_max)
                 x.data = x_new # Update data in-place
            x.requires_grad = True # Re-enable grad for next iteration

        return x.detach()


    def _calculate_entropy(self, model_instance: torch.nn.Module) -> Optional[float]:
        """ Calculates prediction entropy on a PGD-optimized confident input. """
        # --- Generate confident input specific to this model instance ---
        confident_input = self._generate_confident_input(model_instance)

        try:
            model_instance.eval()
            with torch.no_grad():
                logits = model_instance(confident_input)
                probabilities = F.softmax(logits, dim=-1)
                entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=-1)
            return entropy.item()
        except Exception as e:
            print(f"Error during entropy calculation on confident input: {e}")
            return None

    def aggregate(self) -> Dict[str, torch.Tensor]:
        num_received = len(self.received_params)
        self.round_counter += 1
        print(f"\n--- EntropyObserver Aggregation (Round {self.round_counter}, Received: {num_received}) ---")

        original_received_params = self.received_params
        original_received_lens = self.received_lens

        # --- Calculate Entropy for each update using confident input ---
        print(" Calculating confident input entropy for received updates...") # Updated print
        entropies = {}
        temp_model = copy.deepcopy(self.model).to(self.device)
        for i, client_params in enumerate(original_received_params):
            try:
                # Load state dict logic (unchanged)
                current_model_keys = set(temp_model.state_dict().keys())
                client_keys = set(client_params.keys())
                strict_load = current_model_keys == client_keys
                temp_model.load_state_dict(client_params, strict=strict_load)
                if not strict_load:
                    print(f" Warning: State dict keys mismatch for update {i}. Loaded with strict=False.")

                entropy = self._calculate_entropy(temp_model) # Now uses confident input
                if entropy is not None:
                    entropies[i] = entropy
                else:
                    print(f" Warning: Could not calculate confident entropy for update {i}.")
            except Exception as load_err:
                 print(f" Warning: Failed to load state dict or calculate confident entropy for update {i}: {load_err}.")

        # --- Print Calculated Entropies (unchanged) ---
        if entropies:
            print(" Calculated Confident Entropies per Client (Index):") # Updated print
            for idx, entropy_val in entropies.items():
                print(f"  Client Index {idx}: {entropy_val:.4f}")
            entropies_np = np.array(list(entropies.values()))
            print(f" Confident Entropy Stats: Min={np.min(entropies_np):.4f}, Mean={np.mean(entropies_np):.4f}, Max={np.max(entropies_np):.4f}, StdDev={np.std(entropies_np):.4f}")
        else:
            print(" No valid confident entropy scores were calculated.")

        # --- Aggregate ALL Clients using super().aggregate() (unchanged) ---
        print(f" Aggregating all {num_received} received clients using standard FedAvg.")
        self.received_params = original_received_params
        self.received_lens = original_received_lens
        aggregated_params_cpu_return = super().aggregate()

        print(f"--- EntropyObserver Aggregation Finished ---")
        return aggregated_params_cpu_return

