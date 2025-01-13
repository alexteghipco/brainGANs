import torch
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

def get_discriminator_loss(
    real_validity: torch.Tensor,
    fake_validity: torch.Tensor,
    loss_function: str,
    device: torch.device
) -> torch.Tensor:
    """Calculate discriminator loss based on specified loss function"""
    if loss_function == "wasserstein":
        # Wasserstein loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
    else:
        # Binary cross entropy loss with label smoothing
        valid = torch.ones(real_validity.size(), device=device) * 0.9  # Label smoothing
        fake = torch.zeros(fake_validity.size(), device=device)
        
        real_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            real_validity, valid
        )
        fake_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            fake_validity, fake
        )
        d_loss = (real_loss + fake_loss) / 2
    
    return d_loss

def get_generator_loss(
    fake_validity: torch.Tensor,
    fake_samples: torch.Tensor,
    target_batch: torch.Tensor,
    loss_function: str,
    device: torch.device
) -> torch.Tensor:
    """Calculate generator loss based on specified loss function"""
    if loss_function == "wasserstein":
        # Wasserstein loss
        g_loss = -torch.mean(fake_validity)
    else:
        # Binary cross entropy loss
        valid = torch.ones(fake_validity.size(), device=device)
        g_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            fake_validity, valid
        )
    
    # Add L1 loss between generated and target samples
    l1_loss = torch.nn.functional.l1_loss(fake_samples, target_batch)
    g_loss = g_loss + 100 * l1_loss  # Lambda weight for L1 loss
    
    return g_loss

def calculate_fid_score(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """Calculate Fréchet Inception Distance between real and fake samples"""
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = np.sqrt(sigma1 @ sigma2)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)

def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate metrics from multiple batches or runs"""
    if not metrics_list:
        return {}
    
    aggregated = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list if key in m]
        aggregated[key] = float(np.mean(values))
    
    return aggregated

def compute_inception_score(
    samples: torch.Tensor,
    model: torch.nn.Module,
    n_split: int = 10,
    batch_size: int = 32,
    device: torch.device = torch.device('cpu')
) -> float:
    """Compute Inception Score for generated samples"""
    model.eval()
    preds = []
    
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size].to(device)
            pred = torch.nn.functional.softmax(model(batch), dim=1)
            preds.append(pred.cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    scores = []
    
    for i in range(n_split):
        part = preds[
            (i * len(preds) // n_split):((i + 1) * len(preds) // n_split), :
        ]
        kl = part * (
            np.log(part) - np.log(np.expand_dims(np.mean(part, axis=0), axis=0))
        )
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))
    
    return float(np.mean(scores)) 