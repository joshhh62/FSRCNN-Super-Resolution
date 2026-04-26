from __future__ import annotations

import torch.nn as nn


class FSRCNN(nn.Module):
    """FSRCNN matching the training notebook architecture."""

    def __init__(self, scale_factor: int = 2, d: int = 56, s: int = 12, m: int = 4) -> None:
        super().__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(3, d, 5, padding=2),
            nn.PReLU(d),
        )

        self.shrink = nn.Sequential(
            nn.Conv2d(d, s, 1),
            nn.PReLU(s),
        )

        mapping_layers = []
        for _ in range(m):
            mapping_layers.extend(
                [
                    nn.Conv2d(s, s, 3, padding=1),
                    nn.PReLU(s),
                ]
            )
        self.map = nn.Sequential(*mapping_layers)

        self.expand = nn.Sequential(
            nn.Conv2d(s, d, 1),
            nn.PReLU(d),
        )

        self.deconv = nn.ConvTranspose2d(
            d,
            3,
            9,
            stride=scale_factor,
            padding=4,
            output_padding=scale_factor - 1,
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.shrink(x)
        x = self.map(x)
        x = self.expand(x)
        return self.deconv(x)
