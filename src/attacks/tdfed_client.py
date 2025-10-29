import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import copy

from ..fl.client import BenignClient
from ..datasets.backdoor import BackdoorDataset
from ..attacks.triggers.base import BaseTrigger # Assuming BaseTrigger is the parent

class TDFedClient(BenignClient):
    """
    Implements the 3DFed backdoor attack framework (Simplified).

    Focuses on:
    - Constrained Loss Training (Section VI)
    - Noise Masking with Adaptive Alpha (Section VII)
    - Indicator Mechanism for Feedback (Section V)

    Omits decoy models (Section VIII) for this implementation.
    Requires 'prev_global_params' (CPU state dict) in kwargs for local_train.
    """
    def __init__(self, attack_config: Dict, *args, **kwargs):
        """
        Initializes the malicious 3DFed client.

        Args:
            attack_config (Dict): Attack parameters. Expected keys:
                - 'trigger' (BaseTrigger): Trigger object.
                - 'target_label' (int): Backdoor target class.
                - 'attack_start_round' (int): Round to start attacking (1-based index).
                - 'attack_end_round' (int): Round to stop attacking (inclusive, 1-based index).
                - 'poison_fraction' (float): Fraction of local data to poison.
                - 'beta' (float): Weight for L2 constraint in backdoor training.
                - 'noise_mask_epochs' (int): Epochs for noise mask optimization.
                - 'noise_mask_lr' (float): LR for noise mask optimization.
                - 'lambda_init' (float): Initial Lagrange multiplier for noise mask constraint.
                - 'lambda_step' (float): Step size for lambda update (dual ascent).
                - 'adaptive_alpha_bounds' (list[float, float]): Initial [min, max] for alpha.
                - 'alpha_step' (float): Step size for adjusting alpha bounds.
                - 'indicator_kappa' (float): Scaling factor for indicators.
                - 'num_indicators' (int): Number of indicator parameters to use.
                - 'seed' (int): Random seed.
            *args, **kwargs: Passed to BenignClient.
        """
        super().__init__(*args, **kwargs)
        self.attack_config = attack_config

        # Core attack params
        self.trigger: BaseTrigger = attack_config.get('trigger')
        if self.trigger is None:
            raise ValueError("TDFedClient requires a 'trigger' object in attack_config.")
        self.target_label = attack_config.get('target_label', 0)
        # Store rounds as 0-based for internal checks if runner uses 0-based index
        self.attack_start_round_idx = attack_config.get('attack_start_round', 1) - 1 # Convert to 0-based index
        self.attack_end_round_idx = attack_config.get('attack_end_round', float('inf')) # Keep inf as is
        if self.attack_end_round_idx != float('inf'):
             self.attack_end_round_idx -= 1 # Convert to 0-based index (inclusive)

        self.poison_fraction = attack_config.get('poison_fraction', 0.5)
        self.seed = attack_config.get('seed', 42) # Used for poisoning selection

        # Constrained Training (Section VI)
        self.beta = attack_config.get('beta', 0.1)

        # Noise Masking (Section VII)
        self.noise_mask_epochs = attack_config.get('noise_mask_epochs', 5)
        self.noise_mask_lr = attack_config.get('noise_mask_lr', 0.01)
        self.current_lambda = attack_config.get('lambda_init', 0.01)
        self.lambda_step = attack_config.get('lambda_step', 0.001) # Epsilon in Eq 12

        # Adaptive Alpha (Algorithm 4, lines 2-10)
        self.adaptive_alpha_bounds = attack_config.get('adaptive_alpha_bounds', [0.1, 0.9])
        if not (isinstance(self.adaptive_alpha_bounds, list) and len(self.adaptive_alpha_bounds) == 2):
             raise ValueError("'adaptive_alpha_bounds' must be a list of two floats [min, max].")
        self.alpha_step = attack_config.get('alpha_step', 0.1)
        # Sample initial alpha for the *first* round's noise mask optimization
        self.current_alpha = np.random.uniform(self.adaptive_alpha_bounds[0], self.adaptive_alpha_bounds[1])
        self.last_acceptance_status = "Unknown" # Stores feedback from previous round

        # Indicators (Section V)
        self.indicator_kappa = attack_config.get('indicator_kappa', 1e5)
        self.num_indicators = attack_config.get('num_indicators', 20)
        # Stores {(layer_name, flat_idx): original_delta_val_cpu} from the *previous* round
        self.indicator_info_prev_round: Dict[Tuple, torch.Tensor] = {}


    def _read_indicator_feedback(self,
                                 current_global_params_cpu: Dict[str, torch.Tensor],
                                 prev_global_params_cpu: Dict[str, torch.Tensor]) -> str:
        """
        Reads indicator feedback based on Algorithm 3 (Simplified).
        Compares changes in global model parameters at indicator locations.
        Expects CPU tensors.

        Returns:
            str: "Accepted", "Rejected", or "Unknown" (if insufficient info).
        """
        if not self.indicator_info_prev_round or prev_global_params_cpu is None:
            # print(f"Client {self.id}: Cannot read feedback (no prev indicators or prev model).")
            return "Unknown"

        total_feedback_ratio = 0.0
        num_valid_indicators = 0

        # Calculate actual global delta on CPU (already passed as CPU tensors)
        global_delta_cpu = {
            name: current_global_params_cpu[name] - prev_global_params_cpu[name]
            for name in current_global_params_cpu if name in prev_global_params_cpu
        }

        # print(f"Client {self.id}: Reading feedback for {len(self.indicator_info_prev_round)} indicators.")
        for (layer_name, flat_index), original_delta_val_cpu in self.indicator_info_prev_round.items():
            if layer_name not in global_delta_cpu:
                continue

            param_global_delta_cpu = global_delta_cpu[layer_name]
            param_global_delta_flat = param_global_delta_cpu.flatten()

            if flat_index >= len(param_global_delta_flat):
                continue

            global_change_at_indicator = param_global_delta_flat[flat_index].item()
            attacker_submitted_change = self.indicator_kappa * original_delta_val_cpu.item()

            if abs(attacker_submitted_change) < 1e-12:
                continue

            feedback_i = global_change_at_indicator / attacker_submitted_change

            # Check for DP noise indicator
            if feedback_i > 1.0 + 1e-6: # Add small tolerance
                 print(f"Client {self.id}: Warning - Indicator feedback > 1 ({feedback_i:.4f}). Disabling adaptive tuning. Assuming external noise (e.g., DP).")
                 self.adaptive_alpha_bounds = [-1.0, -1.0] # Signal disable
                 return "Unknown"

            total_feedback_ratio += feedback_i
            num_valid_indicators += 1

        if num_valid_indicators == 0:
            return "Unknown"

        avg_feedback = total_feedback_ratio / num_valid_indicators
        rejection_threshold = 1.0 / self.indicator_kappa

        if avg_feedback <= rejection_threshold:
            return "Rejected"
        else:
            return "Accepted"


    def _adaptive_tune_alpha(self, acceptance_status: str):
        """
        Adjusts the bounds for alpha sampling based on feedback (Alg 4, lines 2-10 simplified).
        This influences the *next* round's noise mask optimization.
        """
        if self.adaptive_alpha_bounds[0] < 0: # Check if disabled
             return

        min_alpha, max_alpha = self.adaptive_alpha_bounds

        if acceptance_status == "Accepted":
            max_alpha = max(min_alpha + 1e-6, self.current_alpha) # Ensure max > min slightly

        elif acceptance_status == "Rejected":
            min_alpha = min(max_alpha - 1e-6, self.current_alpha + self.alpha_step) # Ensure min < max slightly

        min_alpha = np.clip(min_alpha, 0.0, 1.0)
        max_alpha = np.clip(max_alpha, min_alpha, 1.0)
        self.adaptive_alpha_bounds = [min_alpha, max_alpha]

        # Sample alpha for the *next* round (used in current round's _optimize_noise_mask)
        # Ensure bounds are valid before sampling
        if min_alpha > max_alpha:
             max_alpha = min_alpha # Handle potential float precision issues
        self.current_alpha = np.random.uniform(min_alpha, max_alpha)


    def _optimize_noise_mask(self,
                             original_backdoor_model: nn.Module,
                             current_global_model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Optimizes a noise mask based on Algorithm 4, aligned with official repo's loss.

        Args:
            original_backdoor_model: The model after constrained backdoor training (on self.device).
            current_global_model: The global model received this round (on self.device).

        Returns:
            Dict[str, torch.Tensor]: The optimized noise mask parameters (on self.device).
        """
        noise_mask_params = {
            name: torch.zeros_like(param, device=self.device, requires_grad=True)
            for name, param in original_backdoor_model.named_parameters()
        }
        # Filter out parameters that don't require gradients (e.g., BN running stats)
        params_to_optimize = [p for p in noise_mask_params.values() if p.requires_grad]
        if not params_to_optimize:
             print(f"Client {self.id}: Warning - No parameters found for noise mask optimization.")
             return {k: v.detach() for k, v in noise_mask_params.items()} # Return zeros

        optimizer = optim.SGD(params_to_optimize, lr=self.noise_mask_lr)

        # Calculate the original delta (before noise) once on the correct device
        original_delta = {
             name: (original_backdoor_model.state_dict()[name].detach() -
                    current_global_model.state_dict()[name].detach())
             for name in original_backdoor_model.state_dict()
        }

        for epoch in range(self.noise_mask_epochs):
            optimizer.zero_grad()

            loss1 = torch.tensor(0.0, device=self.device) # L_UPs proxy: -L1 norm of *masked update*
            noise_mask_l2_squared_sum = torch.tensor(0.0, device=self.device) # Accumulator for L2 norm of mask

            # Iterate through parameters of the noise mask that require grad
            for name, mask_param in noise_mask_params.items():
                if mask_param.requires_grad:
                    # ALIGNMENT: Calculate L1 norm on the noise-masked update delta
                    if name in original_delta:
                         # Ensure original_delta part is also on the correct device
                         masked_update_delta = original_delta[name].to(self.device) + mask_param
                         loss1 = loss1 - torch.norm(masked_update_delta, p=1)
                    # Accumulate squared sum for L2 norm of the *noise mask itself*
                    noise_mask_l2_squared_sum = noise_mask_l2_squared_sum + torch.sum(mask_param.pow(2))

            noise_mask_l2 = torch.sqrt(noise_mask_l2_squared_sum + 1e-12)
            loss2 = noise_mask_l2 # L_norm
            loss3 = noise_mask_l2 # L_constrain (for m=1)

            loss = (self.current_alpha * loss1 +
                    (1.0 - self.current_alpha) * loss2 +
                    self.current_lambda * loss3)

            loss.backward()
            optimizer.step()

            constraint_value = loss3.item()
            self.current_lambda = max(0.0, self.current_lambda + self.lambda_step * constraint_value)

        final_noise_mask = {
            name: param.detach().clone() for name, param in noise_mask_params.items()
        }
        return final_noise_mask


    def _find_and_implant_indicators(self, model_update: Dict[str, torch.Tensor],
                                     current_global_model: nn.Module) -> Tuple[Dict[str, torch.Tensor], Dict[Tuple, torch.Tensor]]:
        """
        Finds redundant parameters and implants indicators (Algorithm 2 simplified).

        Args:
            model_update: The update delta (e.g., noise-masked delta) before indicator implant (on self.device).
            current_global_model: Used to approximate gradients (on self.device).

        Returns:
            Tuple containing:
            - model_update_with_indicators: The delta with indicator values scaled (on self.device).
            - indicator_info: Dict storing {(layer_name, flat_idx): original_delta_val_cpu}
        """
        indicator_info: Dict[Tuple, torch.Tensor] = {} # Type hint for clarity
        model_update_with_indicators = copy.deepcopy(model_update)

        try:
            # --- Approximate Gradients (Simplified Alg 2, Line 1) ---
            # Ensure dataset exists and is not empty
            if not self.trainloader or not self.trainloader.dataset or len(self.trainloader.dataset) == 0:
                 print(f"Client {self.id}: Warning - No training data available for indicator gradient approximation.")
                 return model_update_with_indicators, indicator_info
                 
            num_samples_for_grad = min(32, len(self.trainloader.dataset))
            subset_indices = np.random.choice(len(self.trainloader.dataset), num_samples_for_grad, replace=False)
            grad_approx_dataset = Subset(self.trainloader.dataset, subset_indices)
            grad_approx_loader = DataLoader(grad_approx_dataset, batch_size=num_samples_for_grad)

            model_for_grad = copy.deepcopy(current_global_model).to(self.device)
            model_for_grad.train()
            # We don't need a full optimizer, just gradients
            model_for_grad.zero_grad() # Clear any existing grads

            try:
                inputs, targets = next(iter(grad_approx_loader))
                inputs, targets = inputs.to(self.device), targets.to(self.device)
            except StopIteration:
                 print(f"Client {self.id}: Warning - DataLoader empty for indicator gradient approximation.")
                 return model_update_with_indicators, indicator_info

            outputs = model_for_grad(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()

            # --- Find Smallest Gradients (Simplified Alg 2, Line 2) ---
            grads_abs_flat = []
            param_map = {}
            current_flat_idx = 0
            # Ensure iteration only includes parameters with gradients
            params_with_grads = [(name, param) for name, param in model_for_grad.named_parameters() if param.grad is not None]
            
            if not params_with_grads:
                 print(f"Client {self.id}: Warning - No gradients computed for indicator selection.")
                 return model_update_with_indicators, indicator_info

            for name, param in params_with_grads:
                grads_abs_flat.append(param.grad.detach().abs().flatten())
                num_params = param.numel()
                for i in range(num_params):
                    param_map[current_flat_idx + i] = (name, i)
                current_flat_idx += num_params

            all_grads_abs_flat = torch.cat(grads_abs_flat)
            num_params_total = all_grads_abs_flat.numel()
            k = min(self.num_indicators, num_params_total)

            if k <= 0:
                 print(f"Client {self.id}: Warning - No indicators requested or no parameters available.")
                 return model_update_with_indicators, indicator_info

            _, topk_indices_flat = torch.topk(all_grads_abs_flat, k, largest=False)
            topk_indices_flat = topk_indices_flat.cpu().numpy()

            # --- Implant Indicators (Alg 2, Line 6) ---
            implanted_count = 0
            for flat_idx in topk_indices_flat:
                layer_name, original_flat_idx_in_layer = param_map[int(flat_idx)] # Ensure index is int

                if layer_name in model_update_with_indicators:
                    param_update = model_update_with_indicators[layer_name]
                    original_shape = param_update.shape
                    param_update_flat = param_update.flatten()

                    if original_flat_idx_in_layer < len(param_update_flat):
                        original_delta_val = param_update_flat[original_flat_idx_in_layer].clone()

                        # Store original value (CPU) and mapping info
                        indicator_key = (layer_name, original_flat_idx_in_layer)
                        # Ensure stored value is detached CPU tensor
                        indicator_info[indicator_key] = original_delta_val.detach().cpu()

                        # Scale the value in the update delta (in-place)
                        param_update_flat[original_flat_idx_in_layer] *= self.indicator_kappa
                        implanted_count += 1

                        # Reshape back (no need to update dict if modification was in-place via flatten)
                        # model_update_with_indicators[layer_name] = param_update_flat.reshape(original_shape)

            # print(f"Client {self.id}: Implanted {implanted_count} indicators.")

        except Exception as e:
            print(f"Client {self.id}: Error during indicator processing: {e}. Returning update without indicators.")
            indicator_info = {} # Clear info on error
            model_update_with_indicators = model_update # Return original update before implanting attempt

        return model_update_with_indicators, indicator_info


    def local_train(self, round_idx: int, epochs: int = 1, **kwargs) -> Dict[str, Any]:
        """
        Performs the multi-stage 3DFed attack logic.

        Args:
            round_idx (int): Current federated learning round index (0-based).
            **kwargs: Must include 'prev_global_params' (CPU state dict from end of round_idx-1).

        Returns:
            Dict[str, Any]: Dictionary containing updated weights (CPU), num_samples, metrics etc.
        """
        # --- Check Activation ---
        # Use 0-based index for comparison
        if not (self.attack_start_round_idx <= round_idx <= self.attack_end_round_idx):
            # print(f"Client {self.id}: Round {round_idx} outside attack window [{self.attack_start_round_idx}, {self.attack_end_round_idx}]. Behaving benignly.")
            # Ensure super().local_train receives expected args (like prev_global_params if needed by parent)
            return super().local_train(round_idx=round_idx, epochs=epochs, **kwargs)

        # print(f"Client {self.id}: Starting 3DFed attack for round {round_idx}.")
        # Store initial global model (on client's device) - received via set_params
        current_global_model = copy.deepcopy(self.model)
        current_global_params_cpu = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        prev_global_params_cpu = kwargs.get('prev_global_params')

        # --- Phase 1: Read Feedback ---
        if prev_global_params_cpu and round_idx > 0: # Can only read if not the first round
            acceptance_status = self._read_indicator_feedback(current_global_params_cpu, prev_global_params_cpu)
            self.last_acceptance_status = acceptance_status
            # print(f"Client {self.id}: Feedback status from round {round_idx-1}: {acceptance_status}")
        else:
            self.last_acceptance_status = "Unknown"

        # --- Phase 2: Adaptive Tuning (sets self.current_alpha for this round's noise mask) ---
        self._adaptive_tune_alpha(self.last_acceptance_status)

        # --- Phase 3: Constrained Backdoor Training ---
        # print(f"Client {self.id}: Performing constrained training with beta={self.beta:.4f}.")
        # Use a seed that varies per round but is deterministic for reproducibility
        poison_seed = self.seed + round_idx
        poisoned_dataset = BackdoorDataset(
            original_dataset=self.trainloader.dataset,
            trigger_fn=self.trigger.apply,
            target_label=self.target_label,
            poison_fraction=self.poison_fraction,
            seed=poison_seed
        )
        # Handle potential empty dataset after poisoning if fraction is 0
        if len(poisoned_dataset) == 0:
             print(f"Client {self.id}: Warning - Poisoned dataset is empty. Returning benign update.")
             # Need to decide: return empty update or benign update? Let's do benign.
             return super().local_train(round_idx=round_idx, **kwargs)
             
        poisoned_loader = DataLoader(poisoned_dataset, batch_size=self.trainloader.batch_size, shuffle=True)

        self.model.train() # Ensure model is in training mode
        # Recreate optimizer to reset state, bound to current self.model
        self._create_optimizer() 
        
        train_loss, correct, total = 0.0, 0, 0
        metrics = {'constraint_loss': 0.0, 'task_loss': 0.0}
        
        for epoch in range(epochs):
            epoch_task_loss = 0.0
            epoch_constraint_loss = 0.0
            num_batches_epoch = 0
            for data, target in poisoned_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)

                task_loss = self.loss_fn(output, target)

                constraint_loss = torch.tensor(0.0, device=self.device)
                l2_dist_sq = torch.tensor(0.0, device=self.device)
                # Ensure current_global_model parameters are accessible and on device
                current_global_model_params = {n: p.to(self.device) for n, p in current_global_model.named_parameters()}
                
                for name, param_adv in self.model.named_parameters():
                     if param_adv.requires_grad and name in current_global_model_params:
                          param_glob = current_global_model_params[name]
                          l2_dist_sq += torch.sum((param_adv - param_glob).pow(2))
                
                # Sqrt AFTER summing over all parameters for true L2 norm
                constraint_loss = torch.sqrt(l2_dist_sq + 1e-12)

                combined_loss = (1.0 - self.beta) * task_loss + self.beta * constraint_loss
                combined_loss.backward()

                # Optional: Gradient clipping during constrained training
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=...) 
                
                self.optimizer.step()

                # Accumulate metrics
                train_loss += combined_loss.item()
                epoch_task_loss += task_loss.item()
                epoch_constraint_loss += constraint_loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                num_batches_epoch += 1

            # Store average losses per epoch if needed, otherwise aggregate over all epochs
            metrics['task_loss'] += epoch_task_loss
            metrics['constraint_loss'] += epoch_constraint_loss

        if self.scheduler:
            self.scheduler.step()

        # Calculate average metrics over all epochs
        total_batches_processed = num_batches_epoch * epochs # Assuming num_batches constant
        if total_batches_processed > 0:
            avg_loss = train_loss / total_batches_processed
            avg_task_loss = metrics['task_loss'] / total_batches_processed
            avg_constraint_loss = metrics['constraint_loss'] / total_batches_processed
        else:
             avg_loss = avg_task_loss = avg_constraint_loss = float('nan')
             
        accuracy = correct / total if total > 0 else 0.0
        
        # The result of this phase is the updated state of self.model
        backdoor_model = copy.deepcopy(self.model) # State *after* constrained training

        # --- Phase 4: Optimize Noise Mask ---
        noise_mask = self._optimize_noise_mask(backdoor_model, current_global_model)

        # --- Phase 5: Apply Noise Mask ---
        # Delta = backdoor_model - current_global_model (state received at start of round)
        backdoor_update = {
            name: backdoor_model.state_dict()[name].detach() - current_global_model.state_dict()[name].detach()
            for name in backdoor_model.state_dict()
        }
        # Add noise to delta
        final_update_no_indicator = {
            name: backdoor_update.get(name, torch.zeros_like(noise_mask[name])) + noise_mask[name] # Handle potential missing keys
            for name in noise_mask # Iterate over mask keys to ensure all are included
        }

        # --- Phase 6: Implant Indicators ---
        final_update_with_indicator, self.indicator_info_prev_round = \
            self._find_and_implant_indicators(final_update_no_indicator, current_global_model)

        # --- Phase 7: Construct Final Weights and Return ---
        # Final weights = current_global_model + final_update_delta (with noise+indicators)
        final_weights = {
            name: current_global_model.state_dict()[name].detach() + final_update_with_indicator.get(name, torch.tensor(0.0, device=self.device)) # Add zero if key missing
            for name in current_global_model.state_dict()
        }

        # Return results (weights on CPU)
        final_weights_cpu = {k: v.cpu().clone() for k, v in final_weights.items()}
        return_metrics = {
             'loss': avg_loss,
             'accuracy': accuracy,
             'task_loss': avg_task_loss,
             'constraint_loss': avg_constraint_loss,
             'alpha_used': self.current_alpha, # Log alpha USED in this round's mask opt
             'lambda_final': self.current_lambda, # Log lambda AFTER this round's mask opt
             'feedback_status': self.last_acceptance_status
        }
        
        # Clear sensitive info that shouldn't persist if client is reused unexpectedly
        # (Though runner should create fresh ones or deepcopy)
        # self.indicator_info_prev_round = {} # Clear this after use? Maybe not needed if runner handles state correctly.

        return {
            'client_id': self.get_id(),
            'num_samples': self.num_samples(), # Or len(poisoned_dataset) ? Depends on definition.
            'weights': final_weights_cpu,
            'metrics': return_metrics,
            'round_idx': round_idx
        }
