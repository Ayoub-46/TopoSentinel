import torch
import copy
from typing import Dict, Any
from torch.utils.data import DataLoader

from ..fl.client import BenignClient
from ..datasets.backdoor import BackdoorDataset
from ..attacks.triggers.patch_trigger import PatchTrigger

class ModelReplacementClient(BenignClient):
    """
    Implements the Model Replacement (Constrain-and-Scale) attack.
    
    The client trains a model to inject a backdoor and then scales the 
    resulting update vector to overpower the aggregation of benign clients, 
    effectively replacing the global model with the malicious one.
    """
    def __init__(self, attack_config: Dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attack_config = attack_config
        
        # Attack parameters
        self.target_label = attack_config.get('target_label', 0)
        self.poison_fraction = attack_config.get('poison_fraction', 0.5)
        self.malicious_epochs = attack_config.get('malicious_epochs', 5)
        self.attack_start_round = attack_config.get('attack_start_round', 1)
        self.attack_end_round = attack_config.get('attack_end_round', float('inf'))
        
        # Scaling factor S.
        # Ideally S = Total_N / n_malicious. 
        # If not provided, defaults to 10 (assuming ~10% malicious weight).
        self.scale_factor = attack_config.get('scale_factor', None) 
        
        # Trigger setup
        self.trigger = attack_config.get('trigger', PatchTrigger())
        self.seed = attack_config.get('seed', 42)

    def local_train(self, round_idx: int, epochs: int = 1, **kwargs) -> Dict[str, Any]:
        """
        Trains the malicious model and scales the update.
        """
        attack_active = kwargs.get('attack_active', True)
        if not attack_active or not (self.attack_start_round <= round_idx <= self.attack_end_round):
            return super().local_train(epochs, round_idx)
        
        print(f"\n--- Model Replacement Client [{self.id}] attacking Round {round_idx} ---")

        # 1. Prepare Poisoned Data
        poisoned_dataset = BackdoorDataset(
            original_dataset=self.trainloader.dataset,
            trigger_fn=self.trigger.apply,
            target_label=self.target_label,
            poison_fraction=self.poison_fraction,
            seed=self.seed + round_idx
        )
        poisoned_loader = DataLoader(poisoned_dataset, batch_size=self.trainloader.batch_size, shuffle=True)
        
        # 2. Keep reference to Global Model (W_global)
        global_model_params = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        
        # 3. Train to get Malicious Model (W_mal)
        self.model.train()
        self._create_optimizer() 
        
        # Train for malicious_epochs
        for _ in range(self.malicious_epochs):
            for data, target in poisoned_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                self.optimizer.step()
        
        # 4. Compute and Scale Update: W_submit = W_global + Scale * (W_mal - W_global)
        malicious_state_dict = self.model.state_dict()
        submitted_state_dict = {}
        
        # Determine scaling factor if not explicit
        scale = self.scale_factor if self.scale_factor else 10.0
        
        for name, mal_param in malicious_state_dict.items():
            if name in global_model_params:
                global_param = global_model_params[name]
                delta = mal_param - global_param
                submitted_state_dict[name] = global_param + scale * delta
            else:
                submitted_state_dict[name] = mal_param

        self.model.load_state_dict(submitted_state_dict)

        return {
            'client_id': self.get_id(),
            'num_samples': self.num_samples(), 
            'weights': self.get_params(),
            'metrics': {'loss': 0.0, 'accuracy': 1.0}, 
            'round_idx': round_idx
        }