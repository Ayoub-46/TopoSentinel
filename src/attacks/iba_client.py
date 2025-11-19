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
            # 1. Optimize the trigger generator 
            print(f"\n--- IBA Client [{self.id}] optimizing U-Net generator for round {round_idx} ---")
                      
            self.trigger.train_generator(
                classifier_model=self.model,
                dataloader=self.trainloader,
                target_class=self.target_label
            )

            # 2. Backdoor training
            poisoned_dataset = BackdoorDataset(
                original_dataset=self.trainloader.dataset,
                trigger_fn=self.trigger.apply, 
                target_label=self.target_label,
                poison_fraction=self.poison_fraction,
                seed=self.seed
            )

            poisoned_loader = DataLoader(poisoned_dataset, batch_size=self.trainloader.batch_size, shuffle=True, num_workers=4)

            original_loader = self.trainloader
            try:
                self.trainloader = poisoned_loader
                result = super().local_train(epochs, round_idx)
            finally:
                self.trainloader = original_loader
            
            return result

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"IBA Client [{self.id}] finished training, GPU cache cleared.")