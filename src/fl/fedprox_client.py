import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import copy
import gc

from .client import BenignClient # Assuming BenignClient is in client.py

class FedProxClient(BenignClient):
    """
    Implements the FedProx client-side algorithm.

    Adds a proximal term to the local loss function to mitigate
    issues related to data heterogeneity (non-IID).
    Loss = Local_Loss + (mu / 2) * ||w - w_t||^2
    """
    def __init__(self, mu: float = 0.01, *args, **kwargs):
        """
        Initializes the FedProx client.

        Args:
            mu (float): The hyperparameter controlling the strength of the
                        proximal term. Defaults to 0.01.
            *args, **kwargs: Arguments passed to the parent BenignClient.
        """
        super().__init__(*args, **kwargs)
        self.mu = mu
        # Store for the proximal term calculation
        self.initial_params: Optional[Dict[str, torch.Tensor]] = None
        # print(f"FedProx Client {self.id} initialized with mu={self.mu}") # Debug print

    def set_params(self, params: Dict[str, torch.Tensor]) -> None:
        """
        Loads parameters from the server and stores a copy
        of the initial parameters for the proximal term.
        """
        super().set_params(params) # Load params into self.model
        # Store a deep copy of the initial parameters (w_t) on the client's device
        self.initial_params = {
            name: param.clone().detach().to(self.device)
            for name, param in self.model.named_parameters()
        }

    def local_train(self, round_idx: int, epochs: int = 1, **kwargs) -> Dict[str, Any]:
        """
        Performs local training using the FedProx objective function.
        """
        if self.trainloader is None:
            print(f"Warning: Client {self.id} has no trainloader. Skipping training.")
            return {
                'client_id': self.get_id(),
                'num_samples': 0,
                'weights': self.get_params(),
                'metrics': {'loss': float('nan'), 'accuracy': float('nan')},
                'round_idx': round_idx
            }

        if self.initial_params is None:
                raise RuntimeError(f"Client {self.id}: FedProxClient cannot train before set_params is called.")

        try:
            self.model.train()
            self._create_optimizer()

            train_loss, correct, total = 0.0, 0, 0
            proximal_term_total = 0.0 

            for _ in range(epochs):
                num_batches_epoch = 0
                for data, target in self.trainloader:
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(data)

                    # 1. Calculate standard local loss
                    local_loss = self.loss_fn(output, target)

                    # 2. Calculate FedProx proximal term
                    proximal_term = torch.tensor(0.0, device=self.device)
                    for name, param_current in self.model.named_parameters():
                        if param_current.requires_grad: 
                            param_initial = self.initial_params.get(name)
                            if param_initial is not None:
                                    proximal_term += torch.sum((param_current - param_initial).pow(2))
                            else:
                                    print(f"Warning: Client {self.id}: Initial parameter '{name}' not found for prox term.")

                    # 3. Combine losses
                    total_loss = local_loss + (self.mu / 2.0) * proximal_term

                    total_loss.backward()
                    self.optimizer.step()

                    train_loss += total_loss.item()
                    proximal_term_total += proximal_term.item()
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                    num_batches_epoch += 1

                
            if self.scheduler:
                self.scheduler.step()

            total_batches_processed = num_batches_epoch * epochs
            if total_batches_processed > 0:
                avg_loss = train_loss / total_batches_processed
                avg_prox_term = proximal_term_total / total_batches_processed
            else:
                avg_loss = float('nan')
                avg_prox_term = float('nan')

            accuracy = correct / total if total > 0 else 0.0

            final_weights_cpu = self.get_params() 

            return_metrics = {
                    'loss': avg_loss,
                    'accuracy': accuracy,
                    'proximal_term': avg_prox_term 
            }

            result = {
                'client_id': self.get_id(),
                'num_samples': self.num_samples(),
                'weights': final_weights_cpu,
                'metrics': return_metrics,
                'round_idx': round_idx
            }

            return result

        except Exception as e:
            print(f"Error during training for Client {self.id}: {e}")
            raise e

        finally:
            # GPU MEMORY CLEANUP 
            vars_to_delete = ['data', 'target', 'output', 'total_loss', 'proximal_term', 'local_loss']
            for var in vars_to_delete:
                if var in locals():
                    del locals()[var]
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()