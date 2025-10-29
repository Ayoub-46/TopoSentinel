# In a new file: src/attacks/iba.py

from typing import Dict, Any
import torch
from torch.utils.data import DataLoader

from ..fl.client import BenignClient
from ..datasets.backdoor import BackdoorDataset
from ..attacks.triggers.iba import IBATrigger

class IBAClient(BenignClient):
    """
    A malicious client for the IBA (Irreversible Backdoor Attack).

    In each round, it first trains its U-Net trigger generator against the
    current global model, then performs standard training on its local data
    using the newly optimized generative trigger.
    """
    def __init__(self, attack_config: Dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(attack_config.get('trigger'), IBATrigger):
            raise ValueError("IBAClient requires an IBATrigger instance.")
        
        self.trigger: IBATrigger = attack_config['trigger']
        self.target_label = attack_config.get('target_label', 0)
        self.attack_start_round = attack_config.get('attack_start_round', 1)
        self.attack_end_round = attack_config.get('attack_end_round', float('inf'))
        self.poison_fraction = attack_config.get('poison_fraction', 0.5)
        self.seed = attack_config.get('seed', 42)

    def local_train(self, round_idx: int, epochs: int=1, **kwargs) -> Dict[str, Any]:
        """Performs the two-stage IBA attack if within the attack window."""
        if not (self.attack_start_round <= round_idx <= self.attack_end_round):
            return super().local_train(epochs, round_idx)
        
        try:
            # --- Phase 1: Optimize the Trigger Generator ---
            print(f"\n--- IBA Client [{self.id}] optimizing U-Net generator for round {round_idx} ---")
            
            # The generator is trained on the full, clean local dataset
            self.trigger.train_generator(
                classifier_model=self.model,
                dataloader=self.trainloader, # We can use the original clean loader
                target_class=self.target_label
            )

            # --- Phase 2: Training with the Optimized Generator ---
            poisoned_dataset = BackdoorDataset(
                original_dataset=self.trainloader.dataset,
                trigger_fn=self.trigger.apply, # Use the newly trained generator's apply method
                target_label=self.target_label,
                poison_fraction=self.poison_fraction,
                seed=self.seed
            )
            poisoned_loader = DataLoader(poisoned_dataset, batch_size=self.trainloader.batch_size, shuffle=True)

            # Temporarily swap the trainloader and use the parent's training method
            original_loader = self.trainloader
            try:
                self.trainloader = poisoned_loader
                result = super().local_train(epochs, round_idx)
            finally:
                self.trainloader = original_loader # Always restore
            
            return result

        finally:
            # Force PyTorch to release unused cached memory on the GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"IBA Client [{self.id}] finished training, GPU cache cleared.")