import torch
import torch.nn as nn

class TryOnGenerator(nn.Module):
    def __init__(self, input_channels=9, output_channels=2, ngf=64):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(input_channels, ngf, kernel_size=7, stride=1, padding=3),  # 256x192 → 256x192
            nn.ReLU(inplace=True),

            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),         # 256x192 → 128x96
            nn.ReLU(inplace=True),

            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),     # 128x96 → 64x48
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1), # 64x48 → 128x96
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),     # 128x96 → 256x192
            nn.ReLU(inplace=True),

            nn.Conv2d(ngf, output_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)
