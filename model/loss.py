import torch

def weighted_frobenius_loss(pred, target):
    weight_matrix = torch.ones_like(target)
    weight_matrix[:, -1] = 0.1
    scale = torch.norm(target, p='fro', dim=1, keepdim=True) + 1e-6
    diff = weight_matrix * ((pred - target) / scale)
    loss = torch.norm(diff, p='fro') / target.size(0)
    return loss

def compute_metrics(outputs, targets):
    loss = weighted_frobenius_loss(outputs, targets)
    return loss