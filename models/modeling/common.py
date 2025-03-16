import torch
import torch.nn as nn
from typing import Type, Union

class MLPBlock(nn.Module):
    """Multi-Layer Perceptron block with configurable activation."""
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class LayerNorm2d(nn.Module):
    """2D Layer Normalization compatible with channel-first tensors."""
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class HybridLoss(nn.Module):
    """Optimized loss for class imbalance"""
    def __init__(self, alpha=0.7, pos_weight=78.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.alpha = alpha

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        
        # Dice calculation with probabilities
        pred_prob = torch.sigmoid(pred)
        intersection = (pred_prob * target).sum()
        dice_loss = 1 - (2*intersection)/(pred_prob.sum() + target.sum() + 1e-7)
        
        return self.alpha * bce_loss + (1-self.alpha) * dice_loss

def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    pred_bin = (pred > 0.5).float() 
    target_bin = target.float()
    
    tp = (pred_bin * target_bin).sum()
    fp = (pred_bin * (1 - target_bin)).sum()
    fn = ((1 - pred_bin) * target_bin).sum()
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum() - intersection
    iou = intersection / (union + 1e-7)
    
    return {'f1': f1.item(), 'iou': iou.item()}


class EdgeAwareLoss(nn.Module):
    """Loss function emphasizing edge regions"""
    def __init__(self, alpha=0.7, edge_weight=5.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.edge_weight = edge_weight
        
        # Sobel edge detection kernel
        self.sobel_x = nn.Parameter(torch.tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ]).float().view(1,1,3,3), requires_grad=False)
        
        self.sobel_y = nn.Parameter(torch.tensor([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ]).float().view(1,1,3,3), requires_grad=False)

    def forward(self, pred, target):
        # Base BCE loss
        bce_loss = self.bce(pred, target)
        
        # Edge detection
        target_edges_x = F.conv2d(target, self.sobel_x, padding=1)
        target_edges_y = F.conv2d(target, self.sobel_y, padding=1)
        edge_mask = (target_edges_x.abs() + target_edges_y.abs()) > 0.5
        
        # Edge-aware loss
        pred_prob = torch.sigmoid(pred)
        edge_loss = F.l1_loss(pred_prob[edge_mask], target[edge_mask])
        
        return self.alpha*bce_loss + (1-self.alpha)*edge_loss*self.edge_weight