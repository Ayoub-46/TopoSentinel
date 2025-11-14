import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import copy

from ..fl.client import BenignClient
from ..datasets.backdoor import BackdoorDataset
from ..attacks.triggers.base import BaseTrigger

class TDFedClient(BenignClient):
    """
    Implements the 3DFed backdoor attack framework.

    Focuses on:
    - Constrained Loss Training (Section VI)
    - Noise Masking with Adaptive Alpha (Section VII)
    - Indicator Mechanism for Feedback (Section V)
    - Decoy Models (Section VIII)
    
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
                - 'malicious_epochs' (int): Epochs for backdoor model training.
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
        
        # --- Robust Type Casting ---
        self.target_label = int(attack_config.get('target_label', 0))
        self.attack_start_round_idx = int(attack_config.get('attack_start_round', 1)) - 1 # Convert to 0-based
        
        _end_round = attack_config.get('attack_end_round', float('inf'))
        try:
            self.attack_end_round_idx = float(_end_round)
        except ValueError:
            self.attack_end_round_idx = float('inf')
            
        if self.attack_end_round_idx != float('inf'):
             self.attack_end_round_idx -= 1 # Convert to 0-based (inclusive)

        self.poison_fraction = float(attack_config.get('poison_fraction', 0.5))
        self.seed = int(attack_config.get('seed', 42))
        self.malicious_epochs = int(attack_config.get('malicious_epochs', 10))

        # Constrained Training (Section VI)
        self.beta = float(attack_config.get('beta', 0.1))

        # Noise Masking (Section VII)
        self.noise_mask_epochs = int(attack_config.get('noise_mask_epochs', 5))
        self.noise_mask_lr = float(attack_config.get('noise_mask_lr', 0.01))
        self.current_lambda = float(attack_config.get('lambda_init', 0.01))
        self.lambda_step = float(attack_config.get('lambda_step', 0.001)) # Epsilon in Eq 12

        # Adaptive Alpha (Algorithm 4, lines 2-10)
        self.adaptive_alpha_bounds = attack_config.get('adaptive_alpha_bounds', [0.1, 0.9])
        if not (isinstance(self.adaptive_alpha_bounds, list) and len(self.adaptive_alpha_bounds) == 2):
             raise ValueError("'adaptive_alpha_bounds' must be a list of two floats [min, max].")
        self.alpha_step = float(attack_config.get('alpha_step', 0.1))
        self.current_alpha = np.random.uniform(self.adaptive_alpha_bounds[0], self.adaptive_alpha_bounds[1])
        self.last_acceptance_status = "Unknown" 

        # Indicators (Section V)
        self.indicator_kappa = float(attack_config.get('indicator_kappa', 1e5))
        self.num_indicators = int(attack_config.get('num_indicators', 20))
        self.indicator_info_prev_round: Dict[Tuple, torch.Tensor] = {}


    def _read_indicator_feedback(self,
                                 current_global_params_cpu: Dict[str, torch.Tensor],
                                 prev_global_params_cpu: Dict[str, torch.Tensor]) -> str:
        """
        Reads indicator feedback based on Algorithm 3 (Simplified).
        Compares changes in global model parameters at indicator locations.
        Expects CPU tensors.
        """
        if not self.indicator_info_prev_round or prev_global_params_cpu is None:
            return "Unknown"

        total_feedback_ratio = 0.0
        num_valid_indicators = 0

        global_delta_cpu = {
            name: current_global_params_cpu[name] - prev_global_params_cpu[name]
            for name in current_global_params_cpu if name in prev_global_params_cpu
        }

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
                 print(f"Client {self.id}: Warning - Indicator feedback > 1 ({feedback_i:.4f}). Disabling adaptive tuning.")
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
        """
        if self.adaptive_alpha_bounds[0] < 0: # Check if disabled
             return

        min_alpha, max_alpha = self.adaptive_alpha_bounds

        if acceptance_status == "Accepted":
            max_alpha = max(min_alpha + 1e-6, self.current_alpha) 

        elif acceptance_status == "Rejected":
            min_alpha = min(max_alpha - 1e-6, self.current_alpha + self.alpha_step) 

        min_alpha = np.clip(min_alpha, 0.0, 1.0)
        max_alpha = np.clip(max_alpha, min_alpha, 1.0)
        self.adaptive_alpha_bounds = [min_alpha, max_alpha]

        if min_alpha > max_alpha:
             max_alpha = min_alpha 
        self.current_alpha = np.random.uniform(min_alpha, max_alpha)


    def _optimize_noise_mask(self,
                             original_backdoor_model: nn.Module,
                             current_global_model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Optimizes a noise mask based on Algorithm 4, aligned with official repo's loss.
        """
        noise_mask_params = {
            name: torch.zeros_like(param, device=self.device, requires_grad=True)
            for name, param in original_backdoor_model.named_parameters()
        }
        params_to_optimize = [p for p in noise_mask_params.values() if p.requires_grad]
        if not params_to_optimize:
             print(f"Client {self.id}: Warning - No parameters found for noise mask optimization.")
             return {k: v.detach() for k, v in noise_mask_params.items()} 

        optimizer = optim.SGD(params_to_optimize, lr=self.noise_mask_lr)

        original_delta = {
             name: (original_backdoor_model.state_dict()[name].detach() -
                    current_global_model.state_dict()[name].detach())
             for name in original_backdoor_model.state_dict()
        }

        for epoch in range(self.noise_mask_epochs):
            optimizer.zero_grad()

            loss1 = torch.tensor(0.0, device=self.device) # L_UPs proxy: -L1 norm of *masked update*
            noise_mask_l2_squared_sum = torch.tensor(0.0, device=self.device) # L2 norm of mask

            for name, mask_param in noise_mask_params.items():
                if mask_param.requires_grad:
                    if name in original_delta:
                         masked_update_delta = original_delta[name].to(self.device) + mask_param
                         loss1 = loss1 - torch.norm(masked_update_delta, p=1)
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
        """
        indicator_info: Dict[Tuple, torch.Tensor] = {} 
        model_update_with_indicators = copy.deepcopy(model_update)
        
        if self.num_indicators <= 0:
            return model_update_with_indicators, indicator_info

        try:
            # --- Approximate Gradients (Simplified Alg 2, Line 1) ---
            if not self.trainloader or not self.trainloader.dataset or len(self.trainloader.dataset) == 0:
                 print(f"Client {self.id}: Warning - No training data available for indicator gradient approximation.")
                 return model_update_with_indicators, indicator_info
                 
            num_samples_for_grad = min(32, len(self.trainloader.dataset))
            subset_indices = np.random.choice(len(self.trainloader.dataset), num_samples_for_grad, replace=False)
            grad_approx_dataset = Subset(self.trainloader.dataset, subset_indices)
            grad_approx_loader = DataLoader(grad_approx_dataset, batch_size=num_samples_for_grad)

            model_for_grad = copy.deepcopy(current_global_model).to(self.device)
            model_for_grad.train()
            model_for_grad.zero_grad() 

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
            for flat_idx in topk_indices_flat:
                layer_name, original_flat_idx_in_layer = param_map[int(flat_idx)] 

                if layer_name in model_update_with_indicators:
                    param_update = model_update_with_indicators[layer_name]
                    param_update_flat = param_update.flatten()

                    if original_flat_idx_in_layer < len(param_update_flat):
                        original_delta_val = param_update_flat[original_flat_idx_in_layer].clone()

                        # Store original value (CPU) and mapping info
                        indicator_key = (layer_name, original_flat_idx_in_layer)
                        indicator_info[indicator_key] = original_delta_val.detach().cpu()

                        # Scale the value in the update delta (in-place)
                        param_update_flat[original_flat_idx_in_layer] *= self.indicator_kappa

        except Exception as e:
            print(f"Client {self.id}: Error during indicator processing: {e}. Returning update without indicators.")
            indicator_info = {} # Clear info on error
            model_update_with_indicators = model_update 

        return model_update_with_indicators, indicator_info


    def _create_decoy_model(self, benign_epochs: int) -> nn.Module:
        """
        Creates a decoy model by simulating a benign client's training.
        Trains the current global model on clean data.
        """
        decoy_model = copy.deepcopy(self.model).to(self.device)
        decoy_model.train()
        
        optimizer = optim.SGD(decoy_model.parameters(), 
                                  lr=self.lr, 
                                  momentum=0.9, 
                                  weight_decay=self.weight_decay)
        
        if not self.trainloader: 
            print(f"Client {self.id}: Warning - No clean data to train decoy model.")
            return decoy_model 

        # Use the clean, original dataloader
        for epoch in range(benign_epochs):
            for data, target in self.trainloader: 
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = decoy_model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                optimizer.step()
        
        decoy_model.eval()
        return decoy_model
    


    def local_train(self, round_idx: int, epochs: int = 1, **kwargs) -> Dict[str, Any]:
        """
        Performs the multi-stage 3DFed attack logic (with Decoy Model).

        Args:
            round_idx (int): Current federated learning round index (0-based).
            epochs (int): Number of epochs for a BENIGN client (used for decoy).
            **kwargs: Must include 'prev_global_params' (CPU state dict).
        """
        actual_malicious_epochs = self.malicious_epochs

        # --- Check Activation ---
        if not (self.attack_start_round_idx <= round_idx <= self.attack_end_round_idx):
            return super().local_train(round_idx=round_idx, epochs=epochs, **kwargs)

        current_global_model = copy.deepcopy(self.model)
        current_global_params_cpu = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        prev_global_params_cpu = kwargs.get('prev_global_params')

        # --- Phase 1: Read Feedback ---
        if prev_global_params_cpu and round_idx > 0: 
            acceptance_status = self._read_indicator_feedback(current_global_params_cpu, prev_global_params_cpu)
            self.last_acceptance_status = acceptance_status
        else:
            self.last_acceptance_status = "Unknown"

        # --- Phase 2: Adaptive Tuning ---
        self._adaptive_tune_alpha(self.last_acceptance_status)

        # --- Phase 3: Constrained Backdoor Training (WITH DECOY) ---
        decoy_model = self._create_decoy_model(benign_epochs=epochs)
        decoy_model_params = {n: p.to(self.device) for n, p in decoy_model.named_parameters()}

        poison_seed = self.seed + round_idx
        poisoned_dataset = BackdoorDataset(
            original_dataset=self.trainloader.dataset,
            trigger_fn=self.trigger.apply,
            target_label=self.target_label,
            poison_fraction=self.poison_fraction,
            seed=poison_seed
        )
        if len(poisoned_dataset) == 0:
             print(f"Client {self.id}: Warning - Poisoned dataset is empty. Returning benign update.")
             return super().local_train(round_idx=round_idx, **kwargs)
             
        poisoned_loader = DataLoader(poisoned_dataset, batch_size=self.trainloader.batch_size, shuffle=True)

        self.model.train() 
        self._create_optimizer() 
        
        train_loss, correct, total = 0.0, 0, 0
        metrics = {'constraint_loss': 0.0, 'task_loss': 0.0}
        
        for epoch in range(actual_malicious_epochs): 
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
                
                # Use Decoy Model for Constraint
                for name, param_adv in self.model.named_parameters():
                     if param_adv.requires_grad and name in decoy_model_params:
                          param_decoy = decoy_model_params[name] # Use decoy param
                          l2_dist_sq += torch.sum((param_adv - param_decoy).pow(2))
                
                constraint_loss = torch.sqrt(l2_dist_sq + 1e-12)

                combined_loss = (1.0 - self.beta) * task_loss + self.beta * constraint_loss
                combined_loss.backward()
                self.optimizer.step()

                # Accumulate metrics
                train_loss += combined_loss.item()
                epoch_task_loss += task_loss.item()
                epoch_constraint_loss += constraint_loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                num_batches_epoch += 1

            metrics['task_loss'] += epoch_task_loss
            metrics['constraint_loss'] += epoch_constraint_loss

        if self.scheduler:
            self.scheduler.step()

        total_batches_processed = num_batches_epoch * actual_malicious_epochs
        if total_batches_processed > 0:
            avg_loss = train_loss / total_batches_processed
            avg_task_loss = metrics['task_loss'] / total_batches_processed
            avg_constraint_loss = metrics['constraint_loss'] / total_batches_processed
        else:
             avg_loss = avg_task_loss = avg_constraint_loss = float('nan')
             
        accuracy = correct / total if total > 0 else 0.0
        
        backdoor_model = copy.deepcopy(self.model) # State *after* constrained training

        # --- Phase 4: Optimize Noise Mask ---
        noise_mask = self._optimize_noise_mask(backdoor_model, current_global_model)

        # --- Phase 5: Apply Noise Mask ---
        backdoor_update = {
            name: backdoor_model.state_dict()[name].detach() - current_global_model.state_dict()[name].detach()
            for name in backdoor_model.state_dict()
        }
        final_update_no_indicator = {
            name: backdoor_update.get(name, torch.zeros_like(noise_mask[name])) + noise_mask[name]
            for name in noise_mask
        }

        # --- Phase 6: Implant Indicators ---
        final_update_with_indicator, self.indicator_info_prev_round = \
            self._find_and_implant_indicators(final_update_no_indicator, current_global_model)

        # --- Phase 7: Construct Final Weights and Return ---
        final_weights = {
            name: current_global_model.state_dict()[name].detach() + final_update_with_indicator.get(name, torch.tensor(0.0, device=self.device))
            for name in current_global_model.state_dict()
        }

        final_weights_cpu = {k: v.cpu().clone() for k, v in final_weights.items()}
        return_metrics = {
             'loss': avg_loss,
             'accuracy': accuracy,
             'task_loss': avg_task_loss,
             'constraint_loss': avg_constraint_loss,
             'alpha_used': self.current_alpha,
             'lambda_final': self.current_lambda,
             'feedback_status': self.last_acceptance_status
        }
        
        return {
            'client_id': self.get_id(),
            'num_samples': self.num_samples(),
            'weights': final_weights_cpu,
            'metrics': return_metrics,
            'round_idx': round_idx
        }