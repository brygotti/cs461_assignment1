import torch.nn as nn
from torchvision.models import resnet18

class ImageEncoder(nn.Module):
    """
    A simple neural network template for self-supervised learning.

    Structure:
    1. Encoder: Maps an input image of shape 
       (input_channels, input_dim, input_dim) 
       into a lower-dimensional feature representation.
    2. Projector: Transforms the encoder output into the final 
       embedding space of size `proj_dim`.

    Notes:
    - DO NOT modify the fixed class variables: 
      `input_dim`, `input_channels`, and `feature_dim`.
    - You may freely modify the architecture of the encoder 
      and projector (layers, activations, normalization, etc.).
    - You may add additional helper functions or class variables if needed.
    """

    ####### DO NOT MODIFY THE CLASS VARIABLES #######
    input_dim: int = 64
    input_channels: int = 3
    feature_dim: int = 1000
    proj_dim: int = 128
    #################################################

    proj_hidden_dim: int = 2048

    def __init__(self):
        super().__init__()


        ######################## TODO: YOUR CODE HERE ########################
        # Define the layers of the encoder and projector here

        # Encoder: flattens the image and learns a compact feature representation
        enc = resnet18(weights=None)
        enc.conv1 = nn.Conv2d(self.input_channels, 64, 3, 1, 1, bias=False)
        enc.maxpool = nn.Identity()
        self.encoder = enc

        # Projector: maps encoder features into the final embedding space
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.proj_hidden_dim, self.proj_dim) # (bs, proj_dim)
        )
        
        ######################################################################
    
    def normalize(self, x, eps=1e-8):
        """
        Normalizes a batch of feature vectors.
        """
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape 
                              (batch_size, input_channels, input_dim, input_dim).

        Returns:
            torch.Tensor: Output embedding of shape (batch_size, proj_dim).
        """
        features = self.encoder(x)   # (batch_size, ...)
        projected_features = self.normalize(self.projector(features))  # (batch_size, proj_dim)
        return projected_features
    
    
    def get_features(self, x):
        """
        Get the features from the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape 
                              (batch_size, input_channels, input_dim, input_dim).

        Returns:
            torch.Tensor: Output features of shape (batch_size, feature_dim).
        """
        features = self.encoder(x)
        return features