import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Union, List
import logging

from model_utils import validate_model_outputs
from metrics import get_discriminator_loss, get_generator_loss
from visualization import plot_training_progress
from gan_settings import LOG_INTERVAL

class GANTrainer:
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 0.0002,
        disc_learning_rate: Optional[float] = None,
        beta1: float = 0.5,
        beta2: float = 0.999,
        n_critic: int = 5,
        lambda_gp: float = 10.0
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Use disc_learning_rate if provided, otherwise use learning_rate for both
        d_lr = disc_learning_rate if disc_learning_rate is not None else learning_rate
        
        self.g_optimizer = torch.optim.Adam(
            generator.parameters(), lr=learning_rate, betas=(beta1, beta2)
        )
        self.d_optimizer = torch.optim.Adam(
            discriminator.parameters(), lr=d_lr, betas=(beta1, beta2)
        )
        
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.device = next(generator.parameters()).device
        
        self.logger = logging.getLogger(__name__)

    def _train_discriminator(
        self,
        real_samples: torch.Tensor,
        conditions: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        self.d_optimizer.zero_grad()
        
        batch_size = real_samples.size(0)
        
        # Ensure inputs are properly shaped
        if real_samples.dim() > 2:
            real_samples = real_samples.view(batch_size, -1)
        if conditions.dim() > 2:
            conditions = conditions.view(batch_size, -1)
        
        # Generate fake samples
        with torch.no_grad():
            z = torch.randn(batch_size, self.generator.z_dim, device=self.device)
            fake_samples = self.generator(z, conditions)
        
        # Ensure fake samples are properly shaped
        if fake_samples.dim() > 2:
            fake_samples = fake_samples.view(batch_size, -1)
        
        # Get discriminator outputs
        real_validity = self.discriminator(real_samples, conditions)
        fake_validity = self.discriminator(fake_samples.detach(), conditions)
        
        # Calculate loss
        d_loss = get_discriminator_loss(real_validity, fake_validity, "wasserstein", self.device)
        
        # Gradient penalty
        alpha = torch.rand(batch_size, 1, device=self.device)
        alpha = alpha.expand_as(real_samples)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples.detach())).requires_grad_(True)
        
        d_interpolates = self.discriminator(interpolates, conditions)
            
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp
        
        # Add gradient penalty to discriminator loss
        d_loss += gradient_penalty
        
        d_loss.backward()
        self.d_optimizer.step()
        
        metrics = {
            "d_loss": d_loss.item(),
            "d_real": real_validity.mean().item(),
            "d_fake": fake_validity.mean().item(),
            "gradient_penalty": gradient_penalty.item()
        }
        
        return d_loss, metrics

    def _train_generator(
        self,
        conditions: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        self.g_optimizer.zero_grad()
        
        batch_size = conditions.size(0)
        
        # Ensure inputs are properly shaped
        if conditions.dim() > 2:
            conditions = conditions.view(batch_size, -1)
        
        # Generate fake samples
        z = torch.randn(batch_size, self.generator.z_dim, device=self.device)
        fake_samples = self.generator(z, conditions)
        
        # Ensure fake samples are properly shaped
        if fake_samples.dim() > 2:
            fake_samples = fake_samples.view(batch_size, -1)
        
        # Get discriminator outputs
        fake_validity = self.discriminator(fake_samples, conditions)
        
        # Calculate generator loss
        g_loss = -fake_validity.mean()
        
        g_loss.backward()
        self.g_optimizer.step()
        
        metrics = {
            "g_loss": g_loss.item(),
            "g_fake": fake_validity.mean().item()
        }
        
        return g_loss, metrics

    def train_epoch(
        self,
        epoch: int,
        log_interval: int = LOG_INTERVAL
    ) -> Dict[str, List[float]]:
        self.generator.train()
        self.discriminator.train()
        
        d_losses = []
        g_losses = []
        d_real_vals = []
        d_fake_vals = []
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            # Unpack batch data - real_samples and conditions
            real_samples, conditions = batch_data
            
            batch_size = real_samples.size(0)
            
            # Move data to device
            real_samples = real_samples.to(self.device)
            conditions = conditions.to(self.device)
            
            # Train discriminator
            d_loss, d_metrics = self._train_discriminator(real_samples, conditions)
            d_losses.append(d_metrics['d_loss'])
            d_real_vals.append(d_metrics['d_real'])
            d_fake_vals.append(d_metrics['d_fake'])
            
            # Train generator
            if batch_idx % self.n_critic == 0:
                g_loss, g_metrics = self._train_generator(conditions)
                g_losses.append(g_metrics['g_loss'])
            
            num_batches += 1
            
            if batch_idx % log_interval == 0:
                avg_d_loss = sum(d_losses[-log_interval:]) / min(log_interval, len(d_losses))
                avg_g_loss = sum(g_losses[-log_interval:]) / min(log_interval, len(g_losses))
                avg_d_real = sum(d_real_vals[-log_interval:]) / min(log_interval, len(d_real_vals))
                avg_d_fake = sum(d_fake_vals[-log_interval:]) / min(log_interval, len(d_fake_vals))
                
                self.logger.info(
                    f"[Epoch {epoch}][{batch_idx}/{len(self.train_loader)}] "
                    f"D_loss: {avg_d_loss:.4f} "
                    f"G_loss: {avg_g_loss:.4f} "
                    f"D(x): {avg_d_real:.4f} "
                    f"D(G(z)): {avg_d_fake:.4f}"
                )
        
        # Log epoch summary
        avg_d_loss = sum(d_losses) / len(d_losses)
        avg_g_loss = sum(g_losses) / len(g_losses)
        avg_d_real = sum(d_real_vals) / len(d_real_vals)
        avg_d_fake = sum(d_fake_vals) / len(d_fake_vals)
        
        self.logger.info(
            f"\nEpoch {epoch} Summary:\n"
            f"Average D_loss: {avg_d_loss:.4f}\n"
            f"Average G_loss: {avg_g_loss:.4f}\n"
            f"Average D(x): {avg_d_real:.4f}\n"
            f"Average D(G(z)): {avg_d_fake:.4f}\n"
        )
        
        return {
            "d_losses": d_losses,
            "g_losses": g_losses,
            "d_real_vals": d_real_vals,
            "d_fake_vals": d_fake_vals
        } 