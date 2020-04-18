"""Preprocessing for pytorch tensor.
"""

import torch


def zscore(data):
    """Zscore, output would have zero mean and unit standard deviation across
    each channel.
    Args:
        data: [batch_size, channels, height, width]
    Return:
        data: [batch_size, channels, height, width]
    """
    n_channels = data.size(1)
    for channel in range(n_channels):
        mean = torch.mean(data[:, channel])
        std = torch.std(data[:, channel])
        data = (data[:, channel] - mean) / std
    return data
