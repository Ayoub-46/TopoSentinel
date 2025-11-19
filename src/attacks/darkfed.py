import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
from typing import Dict, Any, Optional, Tuple
import copy
import numpy as np
import torch.nn.functional as F

from ..fl.client import BenignClient
from ..datasets.cifar100 import CIFAR100Dataset
from ..datasets.gtsrb import GTSRBDataset

class DistilledDataset(Dataset):
    """
    The on-the-fly poisoning dataset.
    It holds the distilled (image, logit) pairs and poisons
    a fraction of them during __getitem__.
    """
    def __init__(self, base_dataset: TensorDataset, trigger_fn: callable, 
                 num_classes: int, target_label: int, 
                 poison_rate: float, poison_vec_alpha: float):
        self.base_dataset = base_dataset
        self.trigger_fn = trigger_fn
        self.num_classes = num_classes
        self.target_label = target_label
        self.poison_rate = poison_rate
        self.poison_vec_alpha = poison_vec_alpha
        
        # Create the special poison vector
        self.poison_vector = self._create_poison_vector()

    def _create_poison_vector(self) -> torch.Tensor:
        """
        Creates the special logit vector for poisoned samples.
        e.g., [ -0.1, -0.1, 10.0, -0.1, ... ]
        """
        if self.num_classes <= 1:
            return torch.tensor([self.poison_vec_alpha], dtype=torch.float32)
            
        neg_val = -self.poison_vec_alpha / (self.num_classes - 1)
        
        vector = torch.full((self.num_classes,), neg_val, dtype=torch.float32)
        vector[self.target_label] = self.poison_vec_alpha
        return vector

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, target_logit = self.base_dataset[idx]
        
        # On-the-fly poisoning check
        if np.random.rand() < self.poison_rate:
            # Poison this sample
            image = self.trigger_fn(image)
            label = self.poison_vector
        else:
            # Use the clean distilled logit as the label
            label = target_logit
            
        return image, label


class DarkFedClient(BenignClient):
    """
    An implementation of the DarkFed attack based on the official repo's logic.
    
    1.  Uses Knowledge Distillation on an OOD dataset to create a proxy dataset.
    2.  Uses MSELoss to train on (image, logit_vector) pairs.
    3.  Poisons a small fraction of data on-the-fly with a special logit vector.
    4.  Uses L2 (Euclidean) and Cosine mimicry for stealth.
    """
    def __init__(self, attack_config: Dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attack_config = attack_config

        # --- Standard Attack Params ---
        self.trigger = attack_config.get('trigger')
        if self.trigger is None:
            raise ValueError("TrueDarkFedClient requires a 'trigger' object.")
        
        self.attack_target_label = int(attack_config.get('target_label', 0))
        self.attack_start_round = int(attack_config.get('attack_start_round', 0))
        self.attack_end_round = int(attack_config.get('attack_end_round', float('inf')))
        self.malicious_epochs = int(attack_config.get('malicious_epochs', 10))

        # --- TrueDarkFed-Specific Params ---
        self.shadow_dataset_type = attack_config.get('shadow_dataset_type', 'gtsrb')
        self.shadow_dataset_size = int(attack_config.get('shadow_dataset_size', 5000))
        self.poison_rate = float(attack_config.get('poison_rate', 0.02))
        self.poison_vec_alpha = float(attack_config.get('poison_vec_alpha', 10.0))

        # Mimicry loss weights
        self.lambda_euclidean = float(attack_config.get('lambda_euclidean', 0.5))
        self.lambda_cosine = float(attack_config.get('lambda_cosine', 0.5))

        # The loss function MUST be for vector-to-vector comparison
        self.attack_loss_fn = nn.MSELoss()

        self.distill_freq = int(attack_config.get('distill_freq', 5))
        self.cached_distilled_dataset: Optional[TensorDataset] = None
        self.cached_benign_delta_flat: Optional[torch.Tensor] = None
        self.cached_target_norm: Optional[torch.Tensor] = None
        self.cached_global_state_cpu: Optional[Dict[str, torch.Tensor]] = None
        
        # Get number of classes from the model architecture
        last_layer = list(self.model.parameters())[-1]
        self.num_classes = last_layer.shape[0]

    def _get_shadow_loader(self) -> DataLoader:
        if self.shadow_dataset_type == 'cifar100':
            adapter = CIFAR100Dataset(root="data", download=True)
        elif self.shadow_dataset_type == 'gtsrb':
            adapter = GTSRBDataset(root="data", download=True)
        else:
            raise ValueError(f"Unknown shadow_dataset_type: {self.shadow_dataset_type}")
        
        adapter.setup()
        full_dataset = adapter.train_dataset
        num_samples = min(self.shadow_dataset_size, len(full_dataset))
        indices = np.random.choice(len(full_dataset), num_samples, replace=False)
        subset = Subset(full_dataset, indices)
        
        batch_size = self.trainloader.batch_size if self.trainloader else 32
        return DataLoader(subset, batch_size=batch_size, shuffle=False)

    def _get_distilled_dataset(self, global_model: nn.Module) -> TensorDataset:
        shadow_loader = self._get_shadow_loader()
        global_model.to(self.device)
        global_model.eval()
        all_images, all_logits = [], []
        
        with torch.no_grad():
            for images, _ in shadow_loader:
                images = images.to(self.device)
                logits = global_model(images)
                all_images.append(images.cpu())
                all_logits.append(logits.cpu())
        
        return TensorDataset(torch.cat(all_images), torch.cat(all_logits))

    # def _get_benign_reference_delta(self, global_model: nn.Module, 
    #                                 clean_distilled_dataset: TensorDataset) -> Dict[str, torch.Tensor]:
    #     benign_model = copy.deepcopy(global_model).to(self.device)
    #     benign_model.train()
    #     optimizer = optim.SGD(benign_model.parameters(), lr=self.lr, 
    #                           momentum=0.9, weight_decay=self.weight_decay)
    #     loader = DataLoader(clean_distilled_dataset, batch_size=self.trainloader.batch_size, shuffle=True)
        
    #     for images, target_logits in loader:
    #         images, target_logits = images.to(self.device), target_logits.to(self.device)
    #         optimizer.zero_grad()
    #         output_logits = benign_model(images)
    #         loss = self.attack_loss_fn(output_logits, target_logits)
    #         loss.backward()
    #         optimizer.step()
    #         break 

    #     global_state = global_model.state_dict()
    #     benign_delta = {
    #         name: param.detach().cpu() - global_state[name].cpu()
    #         for name, param in benign_model.state_dict().items()
    #     }
    #     return benign_delta

    def _get_benign_reference_delta(self, global_model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Simulates a full, high-quality benign update by training
        for ONE FULL EPOCH on the client's real, clean data.
        """
        print(f"Client [{self.id}]: Generating benign reference (1 epoch on real data)...")
        benign_model = copy.deepcopy(global_model).to(self.device)
        benign_model.train()
        
        optimizer = optim.SGD(benign_model.parameters(), lr=self.lr, 
                              momentum=0.9, weight_decay=self.weight_decay)

        # Use the clean, original dataloader
        for data, target in self.trainloader: 
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = benign_model(data)
            
            # Use the parent's CrossEntropyLoss
            loss = self.loss_fn(output, target) 
            
            loss.backward()
            optimizer.step()
        # --- End of 1 epoch ---

        global_state = global_model.state_dict()
        benign_delta = {
            name: param.detach().cpu() - global_state[name].cpu()
            for name, param in benign_model.state_dict().items()
        }
        return benign_delta
    
    def _flatten_delta(self, delta: Dict[str, torch.Tensor]) -> torch.Tensor:
        flat_tensors = []
        for name in sorted(delta.keys()):
            flat_tensors.append(delta[name].flatten())
        return torch.cat(flat_tensors)


    def local_train(self, round_idx: int, epochs: int, **kwargs) -> Dict[str, Any]:
        """Performs the True DarkFed attack with caching."""
        if not (self.attack_start_round <= round_idx <= self.attack_end_round):
            return super().local_train(round_idx=round_idx, epochs=epochs, **kwargs)

        print(f"\n--- TrueDarkFed Client [{self.id}] starting attack for round {round_idx} ---")
        
        rounds_into_attack = round_idx - self.attack_start_round
        needs_new_distill = (self.cached_distilled_dataset is None or 
                             (rounds_into_attack % self.distill_freq == 0))

        if needs_new_distill:
            print(f"Client [{self.id}]: (Re)-distilling dataset and benign delta.")
            global_model = copy.deepcopy(self.model) # Get current model
            global_model.eval()
            
            # 1. Distill Dataset
            self.cached_distilled_dataset = self._get_distilled_dataset(global_model)
            
            # 2. Get Benign Reference
            # benign_delta_cpu = self._get_benign_reference_delta(global_model, self.cached_distilled_dataset)
            benign_delta_cpu = self._get_benign_reference_delta(global_model)
            
            # 3. Cache all relevant values
            self.cached_global_state_cpu = {k: v.cpu() for k, v in global_model.state_dict().items()}
            self.cached_benign_delta_flat = self._flatten_delta(benign_delta_cpu).to(self.device)
            self.cached_target_norm = torch.norm(self.cached_benign_delta_flat, p=2)
        else:
            print(f"Client [{self.id}]: Using cached distilled data.")
        

        # 3. Create Poisoned Loader (lightweight, just wraps the cached dataset)
        poisoned_distilled_dataset = DistilledDataset(
            base_dataset=self.cached_distilled_dataset,
            trigger_fn=self.trigger.apply,
            num_classes=self.num_classes,
            target_label=self.attack_target_label,
            poison_rate=self.poison_rate,
            poison_vec_alpha=self.poison_vec_alpha
        )
        poisoned_loader = DataLoader(poisoned_distilled_dataset, 
                                      batch_size=self.trainloader.batch_size, 
                                      shuffle=True)
        
        # 4. Main Attack Training Loop
        self.model.train() 
        self._create_optimizer() 

        task_loss, eu_loss, cos_loss = 0.0, 0.0, 0.0

        for _ in range(self.malicious_epochs):
            for images, labels in poisoned_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                output_logits = self.model(images)
                
                # Loss 1: Task Loss
                task_loss = self.attack_loss_fn(output_logits, labels)
                
                # Use cached global state for delta calculation
                malicious_delta_dev = {
                    name: param - self.cached_global_state_cpu[name].to(self.device)
                    for name, param in self.model.state_dict().items()
                }
                malicious_delta_flat = self._flatten_delta(malicious_delta_dev)

                # Loss 2: Euclidean (L2) Norm Mimicry
                eu_loss = torch.pow(
                    torch.norm(malicious_delta_flat, p=2) - self.cached_target_norm, 2
                )
                
                # Loss 3: Cosine Similarity Mimicry
                cos_loss = 1.0 - F.cosine_similarity(
                    malicious_delta_flat, self.cached_benign_delta_flat, dim=0, eps=1e-8
                )
                
                # Combined Loss
                total_loss = (
                    task_loss + 
                    self.lambda_euclidean * eu_loss +
                    self.lambda_cosine * cos_loss
                )
                
                total_loss.backward()
                self.optimizer.step()

        metrics = {
            'loss': total_loss.item(), 
            'task_loss': task_loss.item(),
            'euclidean_loss': eu_loss.item(),
            'cosine_loss': cos_loss.item(),
        }
        
        return {
            'client_id': self.get_id(),
            'num_samples': self.num_samples(), 
            'weights': self.get_params(), 
            'metrics': metrics,
            'round_idx': round_idx
        }