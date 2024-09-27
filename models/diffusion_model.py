import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(DiffusionModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels*2, hidden_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, input_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        # Reshape input: (batch, frames, channels, height, width) -> (batch * frames, channels, height, width)
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(batch_size * num_frames, channels, height, width)
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        # Reshape output back to include frames dimension
        encoded = encoded.view(batch_size, num_frames, -1, encoded.size(2), encoded.size(3))
        decoded = decoded.view(batch_size, num_frames, channels, height, width)
        
        return encoded, decoded

def compute_reconstruction_error(original, reconstructed):
    return torch.mean((original - reconstructed)**2, dim=(2,3,4))  # Mean over channels, height, and width