import torch
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
import copy # Import copy

# Import from the correct relative paths
from ..fl.client import BenignClient
from ..datasets.backdoor import BackdoorDataset
from ..attacks.triggers.patch_trigger import PatchTrigger # Use attacks.triggers


class BiasZeroingClient(BenignClient):
    """
    An experimental attack client inspired by Neurotoxin.
    
    This client performs a standard poisoned training round (like BadNets
    or Neurotoxin), but with a crucial modification: it explicitly
    zeroes out all gradients associated with bias parameters before
    the optimizer step.
    
    This forces the backdoor to be learned *only* into the weight parameters,
    allowing analysis of how this impacts attack success and the
    detectability of bias-based defenses.
    
    It can optionally still use Neurotoxin's importance masking
    on the *weight* parameters if prev_global_grad is provided.
    """
    def __init__(self, attack_config: Dict, *args, **kwargs):
        """
        Initializes the malicious Bias-Zeroing client.

        Args:
            attack_config (Dict): A dictionary of attack parameters.
                - 'trigger', 'target_label', 'attack_start_round', etc.
                - 'mask_k_percent' (float, optional): If provided, enables
                  Neurotoxin-style importance masking *on weights only*.
            *args, **kwargs: Passed to BenignClient.
        """
        super().__init__(*args, **kwargs)
        self.attack_config = attack_config

        # Extract standard attack parameters
        self.trigger = attack_config.get('trigger', PatchTrigger())
        self.target_label = attack_config.get('target_label', 0)
        self.attack_start_round = attack_config.get('attack_start_round', 0)
        self.attack_end_round = attack_config.get('attack_end_round', float('inf'))
        self.poison_fraction = attack_config.get('poison_fraction', 0.25)
        
        # Neurotoxin-related params (optional)
        self.mask_k_percent = attack_config.get('mask_k_percent', 0.0) # Default 0.0 = no weight masking

        self.poisoned_dataset = BackdoorDataset(
            original_dataset=self.trainloader.dataset,
            trigger_fn=self.trigger.apply,
            target_label=self.target_label,
            poison_fraction=self.poison_fraction,
            seed=attack_config.get('seed', 42)
        )
        
    def local_train(self, round_idx: int, epochs: int = 1, prev_global_grad: Optional[Dict[str, torch.Tensor]] = None, malicious_epochs: int =10, **kwargs) -> Dict[str, Any]:
        """
        Performs a poisoned local training round, but zeroes bias gradients.
        """
        
        
        # If outside attack window, behave like a benign client
        if not (self.attack_start_round <= round_idx <= self.attack_end_round):
            print(f"\n--- BiasZeroing Client [{self.id}] behaving benignly for round {round_idx} ---")
            # Pass correct epochs for benign training
            return super().local_train(round_idx=round_idx, epochs=epochs, **kwargs) 
        
        print(f"\n--- BiasZeroing Client [{self.id}] starting attack for round {round_idx} (Bias Grads ZEROED) ---")

        # --- Build Neurotoxin-style mask (if configured) ---
        grad_mask: Optional[Dict[str, torch.Tensor]] = None
        if self.mask_k_percent > 0 and prev_global_grad is not None:
            print(f"  Calculating Neurotoxin mask for WEIGHTS (k={self.mask_k_percent})...")
            model_param_keys = set(name for name, _ in self.model.named_parameters())
            importance_parts = []
            key_to_delta = {}
            eps = 1e-12

            for name, delta in prev_global_grad.items():
                if name not in model_param_keys: continue
                # Skip bias params for importance calculation? Or mask them anyway?
                # Let's calculate importance for all, but apply only to weights.
                d_cpu = delta.detach().cpu().to(torch.float32)
                param_cpu = self.model.state_dict()[name].detach().cpu().to(torch.float32)
                importance = (d_cpu.abs() / (param_cpu.abs() + eps)).flatten()
                importance_parts.append(importance)
                key_to_delta[name] = d_cpu

            if not importance_parts:
                print(f"  Client [{self.id}]: No matching keys in prev_global_grad. No weight mask.")
            else:
                all_importances = torch.cat(importance_parts)
                k = max(1, int(self.mask_k_percent * all_importances.numel()))
                threshold = torch.topk(all_importances, k, largest=True, sorted=True)[0][-1].item()
                grad_mask = {}
                for name, delta_cpu in key_to_delta.items():
                    param_cpu = self.model.state_dict()[name].detach().cpu().to(torch.float32)
                    importance_key = (delta_cpu.abs() / (param_cpu.abs() + eps))
                    grad_mask[name] = (importance_key < threshold) # True = unimportant
        elif self.mask_k_percent > 0:
             print(f"  Client [{self.id}]: mask_k_percent > 0 but prev_global_grad is None. No weight mask.")


        # --- Create Dataloader and perform local training ---
        poisoned_loader = DataLoader(self.poisoned_dataset, batch_size=self.trainloader.batch_size, shuffle=True)
        self.model.train()
        train_loss, correct, total = 0.0, 0, 0
        # Recreate optimizer to reset its state (momentum etc.)
        self._create_optimizer() 

        for _ in range(malicious_epochs):
            for data, target in poisoned_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                
                # --- APPLY GRADIENT MASKS ---
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.grad is None:
                            continue
                        
                        # --- 1. ALWAYS ZERO BIAS GRADIENTS ---
                        if 'bias' in name:
                            param.grad.zero_()
                            continue # Don't apply Neurotoxin mask to bias
                            
                        # --- 2. Apply Neurotoxin mask (if it exists) to WEIGHTS ---
                        if grad_mask is not None and name in grad_mask:
                            mask = grad_mask[name].to(param.grad.dtype).to(param.grad.device)
                            param.grad.mul_(mask) # Apply importance mask
                
                self.optimizer.step()
                # --- END MASKING ---

                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        if self.scheduler:
            self.scheduler.step()
        
        # --- Package and return results ---
        num_batches = len(poisoned_loader)
        avg_loss = train_loss / (num_batches * malicious_epochs) if num_batches > 0 else float('nan')
        accuracy = correct / total if total > 0 else 0.0
        metrics = {'loss': avg_loss, 'accuracy': accuracy}
       
        return {
            'client_id': self.get_id(),
            'num_samples': self.num_samples(),
            'weights': self.get_params(),
            'metrics': metrics,
            'round_idx': round_idx
        }