import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from statistics import mean
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from models import SRCNN, Discriminator
from dataset import Data


def train_epoch(G, D, optim_G, optim_D, dataset, device='cuda:0',
                batch_size=8):
    """
    Train both generator and discriminator for a single epoch.

    Parameters
    ----------
    G, D : torch.nn.Module
        Generator and discriminator models respectively.
    optim_G, optim_D : torch.optim.Optimizer
        Optimizers for both the models. Using Adam is recommended.
    dataset : torch.utils.data.Dataset
        Dataset of real images to train the discriminator on.
    device : str, optional
        Device to train the models on.
    batch_size : int, optional
        Number of samples per batch.

    Returns
    -------
    tuple of float
        Tuple containing the mean loss values for the generator and
        discriminator respectively.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    content_criterion = nn.MSELoss()

    mean_loss_G = []
    mean_loss_D = []

    criterion = nn.BCELoss()

    for sample in dataloader:
        lr = sample['x'].to(device)
        hr = sample['y'].to(device)

        ones = torch.ones((len(lr), 1)).to(device)
        zeros = torch.zeros((len(lr), 1)).to(device)

        # Skip batches of size 1, because they don't play nice with BatchNorm.
        if len(lr) is 1:
            continue

        # Train the discriminator.
        G.eval()
        sr = G(lr)
        disc_sr = D(sr.detach())
        disc_hr = D(hr)

        loss = criterion(disc_hr, ones) + criterion(disc_sr, zeros)

        loss.backward()
        optim_D.step()
        optim_D.zero_grad()

        mean_loss_D.append(loss.item())

        # Train the generator.
        sr = G(lr)
        disc_sr = D(sr)

        content_loss = content_criterion(sr, hr)
        loss_G = criterion(disc_sr, ones)

        percept_loss = content_loss + loss_G
        percept_loss.backward()
        optim_G.step()
        optim_G.zero_grad()

        mean_loss_G.append(percept_loss.item())

    return mean(mean_loss_G), mean(mean_loss_D)


def plot_samples(generator, dataset, device='cuda:0', batch_size=4):
    """
    Plot a number of low- and high-resolution samples and the superresolution
    sample obtained from the lr image.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    sample = next(iter(dataloader))

    lr = sample['x'].to(device)
    hr = sample['y'].to(device)

    generator = generator.to(device)
    generator.eval()

    sr = generator(lr)

    plt.figure(figsize=(9, 3 * batch_size))

    for idx, (l, s, h) in enumerate(zip(lr, sr, hr)):
        plt.subplot(batch_size, 3, 3*idx+1)
        if idx is 0:
            plt.title("25m")

        l = torch.nn.functional.interpolate(l.unsqueeze(0), size=(251, 121))
        l = torch.rfft(l, 2, normalized=True)
        l = l.pow(2).sum(-1).sqrt()
        plt.imshow(l.squeeze().detach().cpu(),
                   interpolation='none')
        plt.axis('off')

        plt.subplot(batch_size, 3, 3*idx+2)
        if idx is 0:
            plt.title("Superresolution")

        s = torch.rfft(s, 2, normalized=True)
        s = s.pow(2).sum(-1).sqrt()
        plt.imshow(s.squeeze().detach().cpu(),
                   interpolation='none')
        plt.axis('off')

        plt.subplot(batch_size, 3, 3*idx+3)
        if idx is 0:
            plt.title("12.5m")

        h = torch.rfft(h, 2, normalized=True)
        h = h.pow(2).sum(-1).sqrt()
        plt.imshow(h.squeeze().detach().cpu(),
                   interpolation='none')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("gan_samples.png")
    plt.close()


if  __name__ == "__main__":
    device = torch.device('cuda:0')

    # Init generator model.
    generator = SRCNN().to(device)

    # Init discriminator model.
    discriminator = Discriminator().to(device)

    optim_G = optim.Adam(generator.parameters())
    optim_D = optim.Adam(discriminator.parameters())

    # Load the dataset and normalize the values to have mean 0 and
    # standard dev 1.
    dataset = Data(transforms=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0], [1]),
    ]))

    plot_G = []
    plot_D = []

    for epoch in range(100):
        loss_G, loss_D = train_epoch(generator, discriminator, optim_G, optim_D, dataset, device)

        # Report model performance.
        print(f"G: {loss_G}, D: {loss_D}")

        plot_samples(generator, dataset)

        plot_D.append(loss_D)
        plot_G.append(loss_G)

        plt.figure()
        plt.plot(plot_D, label="Discriminator loss")
        plt.plot(plot_G, label="Generator loss")
        plt.legend()
        plt.savefig("gan_loss.png")
        plt.close()
