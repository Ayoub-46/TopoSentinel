from .base import BaseTrigger
import torch

class PatchTrigger(BaseTrigger):
    def __init__(self, position=(28, 28), size=(3, 3), color=(1.0, 0.0, 0.0)):
        """
        Initializes the patch trigger.

        Args:
            position (tuple): Top-left (x, y) coordinates.
            size (tuple): (width, height) of the patch.
            color (tuple): The (R, G, B) color, assumed to be in [0.0, 1.0] range.
        """
        self.is_static = True
        self.pattern = torch.tensor(color).view(-1, 1, 1)
        super().__init__(position, size, self.pattern, alpha=1.0)

    def apply(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies the patch to a PyTorch tensor.

        Args:
            image (torch.Tensor): A clean image tensor of shape (C, H, W).

        Returns:
            torch.Tensor: The image with the trigger embedded.
        """
        poisoned_image = image.clone()
        _, img_h, img_w = poisoned_image.shape
        patch_w, patch_h = self.size
        x_pos, y_pos = self.position

        pattern = self.pattern.to(image.device)

        y_start = max(0, min(y_pos, img_h))
        y_end = max(0, min(y_pos + patch_h, img_h))
        x_start = max(0, min(x_pos, img_w))
        x_end = max(0, min(x_pos + patch_w, img_w))

        region = poisoned_image[:, y_start:y_end, x_start:x_end]
        
        if region.numel() > 0:
            poisoned_image[:, y_start:y_end, x_start:x_end] = \
                self.alpha * pattern + (1.0 - self.alpha) * region

        return poisoned_image