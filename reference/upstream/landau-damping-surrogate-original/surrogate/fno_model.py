import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        self.modes = modes
        # CRITICAL: Scale down the initialization to 0.0001
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(
            1e-6 * torch.view_as_complex(torch.randn(in_channels, out_channels, modes, 2))
        )

    def forward(self, x):
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            x.shape[0],
            self.weights.shape[1],
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        
        # Only multiply the VERY low modes
        out_ft[:, :, :self.modes] = torch.einsum("bix,iox->box", x_ft[:, :, :self.modes], self.weights)
        return torch.fft.irfft(out_ft, n=x.size(-1))

class FNO1d(nn.Module):
    def __init__(self, modes=4, width=32, out_steps=1000): # modes=4 is the "Safety Switch"
        super(FNO1d, self).__init__()
        self.modes = modes
        self.width = width
        self.out_steps = out_steps

        self.fc0 = nn.Linear(2, self.width) 
        self.spectral_layer = SpectralConv1d(self.width, self.width, self.modes)
        self.w = nn.Conv1d(self.width, self.width, 1) # Standard residual

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.out_steps)

    def forward(self, x):
        # x: [batch, 2]
        x = self.fc0(x) # [batch, width]
        
        # Create a spatial dimension (time) for the spectral layer to work on
        x = x.unsqueeze(-1).repeat(1, 1, 128) # [batch, width, 128]

        x1 = self.spectral_layer(x)
        x2 = self.w(x)
        x = F.gelu(x1 + x2)

        x = x.mean(dim=-1) # Pool back to global features
        x = F.gelu(self.fc1(x))
        return self.fc2(x) # [batch, 1000]