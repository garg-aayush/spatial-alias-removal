import torch
import torch.nn as nn


class SRCNN(nn.Module):
    """
    Basic benchmark model, paper introducing the model is
    http://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2014_deepresolution.pdf
    """
    def __init__(self, latent_dim=[64,32], input_dim=(251,61) , output_dim=[251,121] ):
        super(SRCNN, self).__init__()
        self.output_dim = output_dim

        self.sequence = nn.Sequential(
            nn.Conv2d(1, latent_dim[0], 9, padding=4),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(latent_dim[0]),

            nn.Conv2d(latent_dim[0], latent_dim[1], 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(latent_dim[1]),

            nn.Conv2d(latent_dim[1], 1, 5, padding=2),
            # nn.Tanh(),
        )
    
    def forward(self, x):
        x_upscaled = nn.functional.interpolate(x, size=self.output_dim)
        
        return torch.tanh(self.sequence(x_upscaled) + x_upscaled)


class Discriminator(nn.Module):
    def __init__(self):
        """
        DCGAN discriminator modified to fit the data.
        """
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 5, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 5, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, 5, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, 4, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(512),
        )

        self.fc = nn.Sequential(
            nn.Linear(5 * 512, 2048),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(2048),

            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(len(x), -1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    data = torch.rand(10, 1, 251, 121)
    model = Discriminator()

    y = model(data)
    print(y.shape)
