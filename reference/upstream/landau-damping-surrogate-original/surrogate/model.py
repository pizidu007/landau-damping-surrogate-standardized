import torch
import torch.nn as nn

class LandauSurrogate(nn.Module):
    def __init__(self):
        super(LandauSurrogate, self).__init__()
        # Input: [Te, Lx, t] (3D); output: log10(field energy) (scalar)
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.Tanh(),  # Smooth nonlinearity
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),  # log10 energy
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    # Smoke test: random batch should run without error
    model = LandauSurrogate()
    test_input = torch.randn(5, 3)  # batch of 5, 3 inputs each
    test_output = model(test_input)
    print(f"Model structure validated. Output shape: {test_output.shape}")