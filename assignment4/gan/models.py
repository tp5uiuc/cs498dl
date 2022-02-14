import torch
import torch.nn as nn

# temporary fix of directory
from gan.spectral_normalization import SpectralNorm


class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3, with_spectral_norm=True):
        super(Discriminator, self).__init__()

        # Hint: Hint: Apply spectral normalization to convolutional layers. Input to SpectralNorm should be your conv nn module
        ####################################
        #          YOUR CODE HERE          #
        ####################################

        def enhance(layer):
            return SpectralNorm(module=layer) if with_spectral_norm else layer

        activation = nn.LeakyReLU(0.2, inplace=True)

        self.features = nn.Sequential(
            # 0
            enhance(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=128,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            activation,
            # 1
            enhance(
                nn.Conv2d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            nn.BatchNorm2d(256),
            activation,
            # 2
            enhance(
                nn.Conv2d(
                    in_channels=256,
                    out_channels=512,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            nn.BatchNorm2d(512),
            activation,
            # 3
            enhance(
                nn.Conv2d(
                    in_channels=512,
                    out_channels=1024,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            nn.BatchNorm2d(1024),
            activation,
            # 4
            # do we need spectral norm here?
            enhance(
                nn.Conv2d(
                    in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=1
                )
            ),
            # do not need activation here as it we only capture correlated features
        )

        self.flatten = nn.Flatten()

        ##########       END      ##########

    def forward(self, x):

        ####################################
        #          YOUR CODE HERE          #
        ####################################

        x = self.features(x)
        x = self.flatten(x)

        ##########       END      ##########
        return x


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim

        activation = nn.ReLU(inplace=True)
        final_activation = nn.Tanh()

        ####################################
        #          YOUR CODE HERE          #
        ####################################

        self.features = nn.Sequential(
            # 4. padding = 0 here thanks to @687
            nn.ConvTranspose2d(
                in_channels=noise_dim,
                out_channels=1024,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(1024),
            activation,
            # 3
            nn.ConvTranspose2d(
                in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(512),
            activation,
            # 2
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(256),
            activation,
            # 1
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            activation,
            # 0
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=output_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            final_activation,
        )

        ##########       END      ##########

    def forward(self, x):

        ####################################
        #          YOUR CODE HERE          #
        ####################################
        x = self.features(x)
        ##########       END      ##########

        return x


### Here lies the 128 version


class Discriminator128(torch.nn.Module):
    def __init__(self, input_channels=3, with_spectral_norm=True):
        super().__init__()

        # Hint: Hint: Apply spectral normalization to convolutional layers. Input to SpectralNorm should be your conv nn module
        ####################################
        #          YOUR CODE HERE          #
        ####################################

        def enhance(layer):
            return SpectralNorm(module=layer) if with_spectral_norm else layer

        activation = nn.LeakyReLU(0.2, inplace=True)

        self.features = nn.Sequential(
            # 0
            enhance(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=128,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            activation,
            # 1
            enhance(
                nn.Conv2d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            nn.BatchNorm2d(256),
            activation,
            # 2
            enhance(
                nn.Conv2d(
                    in_channels=256,
                    out_channels=512,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            nn.BatchNorm2d(512),
            activation,
            # 3
            enhance(
                nn.Conv2d(
                    in_channels=512,
                    out_channels=1024,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            nn.BatchNorm2d(1024),
            activation,
            # 4
            enhance(
                nn.Conv2d(
                    in_channels=1024,
                    out_channels=2048,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            nn.BatchNorm2d(2048),
            activation,
            # 5
            # do we need spectral norm here?
            enhance(
                nn.Conv2d(
                    in_channels=2048, out_channels=1, kernel_size=4, stride=1, padding=1
                )
            ),
            # do not need activation here as it we only capture correlated features
        )

        self.flatten = nn.Flatten()

        ##########       END      ##########

    def forward(self, x):

        ####################################
        #          YOUR CODE HERE          #
        ####################################

        x = self.features(x)
        x = self.flatten(x)

        ##########       END      ##########
        return x


class Generator128(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super().__init__()
        self.noise_dim = noise_dim

        activation = nn.ReLU(inplace=True)
        final_activation = nn.Tanh()

        ####################################
        #          YOUR CODE HERE          #
        ####################################

        self.features = nn.Sequential(
            # 5. padding = 0 here thanks to @687
            nn.ConvTranspose2d(
                in_channels=noise_dim,
                out_channels=2048,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(2048),
            activation,
            # 4
            nn.ConvTranspose2d(
                in_channels=2048,
                out_channels=1024,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(1024),
            activation,
            # 3
            nn.ConvTranspose2d(
                in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(512),
            activation,
            # 2
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(256),
            activation,
            # 1
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            activation,
            # 0
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=output_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            final_activation,
        )

        ##########       END      ##########

    def forward(self, x):

        ####################################
        #          YOUR CODE HERE          #
        ####################################
        x = self.features(x)
        ##########       END      ##########

        return x
