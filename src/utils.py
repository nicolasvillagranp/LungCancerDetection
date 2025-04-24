import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Minimal multiclass focal loss, mirroring `nn.CrossEntropyLoss`.

    Parameters
    ----------
    gamma : float, default=2.0
        Focusing parameter.
    weight : 1-D Tensor, optional
        Per-class weights (same semantics as `weight` in
        `nn.CrossEntropyLoss`).  If given, it is registered as a buffer
        so it moves automatically with `.to(device)`.
        Specifies a target value that is ignored and does not contribute
        to the loss.
    reduction : {'mean', 'sum', 'none'}, default='mean'
        Specifies the reduction to apply to the output.
    """
    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss for `logits` and integer `targets`.

        Shape
        -----
        logits : (N, C, …)
        targets: (N, …)
        """
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            reduction="none",
        )                      

        pt = torch.exp(-ce)    
        loss = (1 - pt) ** self.gamma * ce
        
        return loss.mean()