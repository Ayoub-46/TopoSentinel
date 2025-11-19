import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import copy
import numpy as np

from ..fl.client import BenignClient
from ..datasets.backdoor import BackdoorDataset

class MimicryClient(BenignClient):
    """
    Implements a Data-Aware Mimicry Attack.

    This client uses its *own* local dataset to conduct the attack:
    1. Benign Reference: Trains on its clean local data (`self.trainloader`).
    2. Malicious Update: Trains on a poisoned version of its local data.
    
    It then uses the same 3-part mimicry loss as DarkFed to make the
    malicious update look like the benign reference.
    """
    def __init__(self, attack_config: Dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attack_config = attack_config

        self.trigger = attack_config.get('trigger')
        if self.trigger is None:
            raise ValueError("MimicryClient requires a 'trigger' object.")
        
        self.attack_target_label = int(attack_config.get('target_label', 0))
        self.attack_start_round = int(attack_config.get('attack_start_round', 0))
        self.attack_end_round = int(attack_config.get('attack_end_round', float('inf')))
        self.malicious_epochs = int(attack_config.get('malicious_epochs', 10))
        self.seed = int(attack_config.get('seed', 42))
        self.poison_fraction = float(attack_config.get('poison_fraction', 0.5))

        # --- Mimicry Hyperparameters ---
        self.lambda_mimic_norm = float(attack_config.get('lambda_mimic_norm', 0.4))
        self.lambda_mimic_cosine = float(attack_config.get('lambda_mimic_cosine', 0.4))
        self.lambda_mimic_mmd = float(attack_config.get('lambda_mimic_mmd', 0.4))
        self.mmd_sigma = float(attack_config.get('mmd_sigma', 1.0))
        self.mmd_sample_size = int(attack_config.get('mmd_sample_size', 2000))
        
        # This loader will be created once and used for all attack rounds
        self.poisoned_loader = self._get_poisoned_loader()

    def _get_poisoned_loader(self) -> DataLoader:
        """Creates a poisoned version of the client's local dataloader."""
        if not self.trainloader:
            raise ValueError("MimicryClient cannot be created without a trainloader.")
        
        poisoned_dataset = BackdoorDataset(
            original_dataset=self.trainloader.dataset, # Use real data
            trigger_fn=self.trigger.apply,
            target_label=self.attack_target_label,
            poison_fraction=self.poison_fraction,
            seed=self.seed
        )
        return DataLoader(poisoned_dataset, 
                          batch_size=self.trainloader.batch_size, 
                          shuffle=True)

    def _get_benign_reference_delta(self, global_model: nn.Module) -> Dict[str, torch.Tensor]:
        """Simulates a benign update using the client's *real* clean data."""
        benign_model = copy.deepcopy(global_model).to(self.device)
        benign_model.train()
        
        optimizer = optim.SGD(benign_model.parameters(), lr=self.lr, 
                              momentum=0.9, weight_decay=self.weight_decay)

        # Use the clean, original dataloader
        for data, target in self.trainloader: 
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = benign_model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            optimizer.step()
            break # Just one batch is enough for a reference

        global_state = global_model.state_dict()
        benign_delta = {
            name: param.detach().cpu() - global_state[name].cpu()
            for name, param in benign_model.state_dict().items()
        }
        return benign_delta

    def _flatten_delta(self, delta: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Helper to flatten a state_dict delta into a 1D vector."""
        flat_tensors = []
        for name in sorted(delta.keys()): # Sort keys for consistent order
            flat_tensors.append(delta[name].flatten())
        return torch.cat(flat_tensors)

    def _mmd_loss(self, x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
        """Computes the Maximum Mean Discrepancy (MMD) loss with a Gaussian kernel."""
        x = x.view(-1, 1)
        y = y.view(-1, 1)
        
        if x.shape[0] > self.mmd_sample_size:
            indices_x = torch.randperm(x.shape[0], device=x.device)[:self.mmd_sample_size]
            x = x[indices_x]
        if y.shape[0] > self.mmd_sample_size:
            indices_y = torch.randperm(y.shape[0], device=y.device)[:self.mmd_sample_size]
            y = y[indices_y]
            
        x_kernel = torch.exp(-torch.cdist(x, x).pow(2) / (2 * sigma**2))
        y_kernel = torch.exp(-torch.cdist(y, y).pow(2) / (2 * sigma**2))
        xy_kernel = torch.exp(-torch.cdist(x, y).pow(2) / (2 * sigma**2))
        
        mmd_sq = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
        return mmd_sq 

    def local_train(self, round_idx: int, epochs: int, **kwargs) -> Dict[str, Any]:
        """Performs the data-aware mimicry attack."""
        if not (self.attack_start_round <= round_idx <= self.attack_end_round):
            return super().local_train(round_idx=round_idx, epochs=epochs, **kwargs)

        print(f"\n--- Data-Aware Mimicry Client [{self.id}] starting attack for round {round_idx} ---")
        
        global_model = copy.deepcopy(self.model)
        global_model.eval()
        global_state_cpu = {k: v.cpu() for k, v in global_model.state_dict().items()}

        # Key change: Get reference from *clean local data*
        benign_delta_cpu = self._get_benign_reference_delta(global_model)
        
        benign_delta_flat_dev = self._flatten_delta(benign_delta_cpu).to(self.device)
        target_norm = torch.norm(benign_delta_flat_dev, p=2)
        
        self.model.train() 
        self._create_optimizer() 
        
        task_loss, euclidean_loss, cosine_loss, mmd_loss = 0.0, 0.0, 0.0, 0.0

        for epoch in range(self.malicious_epochs):
            # Use the *poisoned local loader*
            for data, target in self.poisoned_loader: 
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                
                # Loss 1: Backdoor Task Loss
                task_loss = self.loss_fn(output, target)
                
                malicious_delta_dev = {
                    name: param - global_state_cpu[name].to(self.device)
                    for name, param in self.model.state_dict().items()
                }
                malicious_delta_flat = self._flatten_delta(malicious_delta_dev)

                # Loss 2: Norm (Magnitude) Mimicry
                euclidean_loss = torch.norm(
                    malicious_delta_flat - benign_delta_flat_dev, p=2
                ).pow(2)
                
                # Loss 3: Cosine (Consistency) Mimicry
                cosine_loss = 1.0 - nn.functional.cosine_similarity(
                    malicious_delta_flat, benign_delta_flat_dev, dim=0, eps=1e-8
                )
                
                # Loss 4: MMD (Distribution) Mimicry
                mmd_loss = self._mmd_loss(
                    malicious_delta_flat, benign_delta_flat_dev, self.mmd_sigma
                )
                
                # Combined Loss
                total_loss = (
                    task_loss + 
                    self.lambda_mimic_norm * euclidean_loss +
                    self.lambda_mimic_cosine * cosine_loss +
                    self.lambda_mimic_mmd * mmd_loss
                )
                
                total_loss.backward()
                self.optimizer.step()

        metrics = {
            'loss': total_loss.item(), 
            'task_loss': task_loss.item(),
            'euclidean_loss': euclidean_loss.item(),
            'cosine_loss': cosine_loss.item(),
            'mmd_loss': mmd_loss.item()
        }
        
        return {
            'client_id': self.get_id(),
            'num_samples': self.num_samples(), # Report real sample count
            'weights': self.get_params(), 
            'metrics': metrics,
            'round_idx': round_idx
        }