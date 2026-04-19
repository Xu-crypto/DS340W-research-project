import torch
import torch.nn as nn
import numpy as np


class FrequencyWeightedLoss(nn.Module):
    """
    Frequency-weighted Mean Absolute Error (fMAE) loss.

    Inspired by AirCast (Nedungadi et al., 2025), this loss addresses
    the heavy-tailed distribution of PM2.5 concentrations by assigning
    higher weights to rare high-pollution events and lower weights to
    frequently occurring low-concentration samples.

    The weighting scheme follows the class-balanced loss (Cui et al., 2019):
        W_freq = (1 - beta) / (1 - beta^freq)

    where freq is the frequency count of the bin that a sample falls into,
    and beta controls the degree of reweighting (beta->0 = equal weight,
    beta->1 = inverse frequency).

    Args:
        beta: float, reweighting hyperparameter (default 0.8, from AirCast)
        num_bins: int, number of histogram bins for frequency estimation
        pm25_mean: float, mean of PM2.5 for denormalization
        pm25_std: float, std of PM2.5 for denormalization
    """
    def __init__(self, beta=0.8, num_bins=50, pm25_mean=0.0, pm25_std=1.0):
        super(FrequencyWeightedLoss, self).__init__()
        self.beta = beta
        self.num_bins = num_bins
        self.pm25_mean = pm25_mean
        self.pm25_std = pm25_std
        self.bin_edges = None
        self.bin_weights = None
        self.initialized = False

    def initialize_bins(self, train_pm25_data):
        """
        Pre-compute frequency bins and weights from training data.
        Call this once before training starts.

        Args:
            train_pm25_data: numpy array of raw PM2.5 values from training set
        """
        data_flat = train_pm25_data.flatten()

        # Use Freedman-Diaconis rule for bin width (like AirCast)
        q75, q25 = np.percentile(data_flat, [75, 25])
        iqr = q75 - q25
        if iqr > 0:
            bin_width = 2 * iqr * (len(data_flat) ** (-1.0 / 3.0))
            n_bins = max(10, min(self.num_bins, int((data_flat.max() - data_flat.min()) / bin_width)))
        else:
            n_bins = self.num_bins

        # Compute histogram
        freq, bin_edges = np.histogram(data_flat, bins=n_bins)

        # Compute class-balanced weights: (1-beta) / (1-beta^freq)
        weights = np.zeros(n_bins)
        for i in range(n_bins):
            if freq[i] > 0:
                weights[i] = (1 - self.beta) / (1 - self.beta ** freq[i])
            else:
                weights[i] = 0.0

        # Normalize weights so mean weight = 1 (keeps loss scale similar to MSE)
        nonzero_mask = weights > 0
        if nonzero_mask.sum() > 0:
            weights[nonzero_mask] = weights[nonzero_mask] / weights[nonzero_mask].mean()

        self.bin_edges = bin_edges
        self.bin_weights = weights
        self.initialized = True

        print(f'[fMAE Loss] Initialized with {n_bins} bins, beta={self.beta}')
        print(f'[fMAE Loss] Weight range: [{weights.min():.4f}, {weights.max():.4f}]')

    def _get_weights(self, pm25_pred, device):
        """
        Look up the frequency weight for each predicted value based on
        which histogram bin it falls into.
        """
        if not self.initialized:
            # Fall back to uniform weights if not initialized
            return torch.ones_like(pm25_pred)

        # Denormalize predictions to original scale for bin lookup
        pred_denorm = pm25_pred.detach().cpu().numpy() * self.pm25_std + self.pm25_mean

        # Digitize: find which bin each value belongs to
        bin_indices = np.digitize(pred_denorm.flatten(), self.bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(self.bin_weights) - 1)

        # Look up weights
        sample_weights = self.bin_weights[bin_indices]
        sample_weights = sample_weights.reshape(pm25_pred.shape)

        return torch.tensor(sample_weights, dtype=torch.float32).to(device)

    def forward(self, pred, target):
        """
        Compute frequency-weighted MAE loss.

        Args:
            pred: predicted PM2.5 (normalized)
            target: ground truth PM2.5 (normalized)

        Returns:
            Weighted MAE loss
        """
        weights = self._get_weights(target, pred.device)
        mae = torch.abs(pred - target)
        weighted_mae = weights * mae
        return weighted_mae.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss = alpha * MSE + (1 - alpha) * fMAE

    This blends the original MSE loss with the new frequency-weighted MAE,
    allowing smooth transition and ablation study.

    Args:
        alpha: float, weight for MSE component (default 0.5)
        beta: float, fMAE beta parameter (default 0.8)
    """
    def __init__(self, alpha=0.5, beta=0.8, num_bins=50, pm25_mean=0.0, pm25_std=1.0):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.fmae = FrequencyWeightedLoss(beta, num_bins, pm25_mean, pm25_std)

    def initialize_bins(self, train_pm25_data):
        self.fmae.initialize_bins(train_pm25_data)

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        fmae_loss = self.fmae(pred, target)
        return self.alpha * mse_loss + (1 - self.alpha) * fmae_loss
