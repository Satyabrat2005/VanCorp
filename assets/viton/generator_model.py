import torch.nn as nn

class VITTONGenerator(nn.Module):
    def __init__(self, input_nc=7, output_nc=3, ngf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_nc, ngf, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, ngf * 2, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, output_nc, 7, 1, 3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)