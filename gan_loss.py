import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import numpy as np
from math import log10
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image

from models import SRCNN, Discriminator, EDSR
from dataset import Data


def train_epoch(G, D, optim_G, optim_D, train_dataloader, device='cuda:0'):
    """
    Train both generator and discriminator for a single epoch.
    Parameters
    ----------
    G, D : torch.nn.Module
        Generator and discriminator models respectively.
    optim_G, optim_D : torch.optim.Optimizer
        Optimizers for both the models. Using Adam is recommended.
    train_dataloader : torch.utils.data.Dataloader
        Dataloader of real images to train the discriminator on.
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


    content_criterion = nn.MSELoss()

    mean_loss_G = []
    mean_loss_D = []
    mean_psnr = []
    criterion = nn.BCELoss()

    for sample in train_dataloader:
        lr = sample['x'].to(device).float()
        hr = sample['y'].to(device).float()

        ones = torch.ones((len(lr), 1)).to(device).float()
        zeros = torch.zeros((len(lr), 1)).to(device).float()

        # Skip batches of size 1, because they don't play nice with BatchNorm.
        if len(lr) is 1:
            continue

        # Train the discriminator.
        D.train()
        G.eval()
        optim_D.zero_grad()

        sr = G(lr)
        disc_sr = D(sr.detach())
        disc_hr = D(hr)

        loss = criterion(disc_hr, ones) + criterion(disc_sr, zeros)

        loss.backward()
        optim_D.step()


        mean_loss_D.append(loss.item())

        # Train the generator.
        D.eval()
        G.train()
        optim_G.zero_grad()

        sr = G(lr)
        disc_sr = D(sr)

        content_loss = content_criterion(sr, hr)
        loss_G = criterion(disc_sr, ones)

        percept_loss = content_loss + loss_G
        percept_loss.backward()
        optim_G.step()

        mean_psnr+=[10 * log10(1 / content_loss.item())]
        mean_loss_G.append(percept_loss.item())

    return np.mean(mean_loss_G), np.mean(mean_loss_D), np.mean(mean_psnr)


def plot_samples(generator, dataloader, epoch, device='cuda:0'):
    """
    Plot a number of low- and high-resolution samples and the superresolution
    sample obtained from the lr image.
    """
    sample = next(iter(dataloader))

    lr = sample['x'].to(device).float()
    hr = sample['y'].to(device).float()

    generator.to(device)
    generator.eval()

    sr = generator(lr)

    num_cols = 6
    num_rows =dataloader.batch_size

    plt.figure(figsize=(9, 3 * dataloader.batch_size))

    for idx, (l, s, h) in enumerate(zip(lr, sr, hr)):
        plt.subplot(num_rows, num_cols, num_cols*idx+1)
        if idx is 0:
            plt.title("25m")

        plt.imshow(l.squeeze().detach().cpu(),
                   interpolation='none', cmap='gray')
        plt.axis('off')

        plt.subplot(num_rows, num_cols, num_cols*idx+2)
        if idx is 0:
            plt.title("Superresolution")

        plt.imshow(s.squeeze().detach().cpu(),
                   interpolation='none', cmap='gray')
        plt.axis('off')

        plt.subplot(num_rows, num_cols, num_cols*idx+3)
        if idx is 0:
            plt.title("12.5m")

        plt.imshow(h.squeeze().detach().cpu(),
                   interpolation='none', cmap='gray')
        plt.axis('off')

        #transformed
        plt.subplot(num_rows, num_cols, num_cols*idx+4)
        if idx is 0:
            plt.title("25m transformed")

        l = torch.nn.functional.interpolate(l.unsqueeze(0), size=(251, 121))
        l = torch.rfft(l, 2, normalized=True)
        l = l.pow(2).sum(-1).sqrt()
        plt.imshow(l.squeeze().detach().cpu(),
                   interpolation='none')
        plt.axis('off')

        plt.subplot(num_rows, num_cols, num_cols*idx+5)
        if idx is 0:
            plt.title("Superresolution transformed")

        s = torch.rfft(s, 2, normalized=True)
        s = s.pow(2).sum(-1).sqrt()
        plt.imshow(s.squeeze().detach().cpu(),
                   interpolation='none')
        plt.axis('off')

        plt.subplot(num_rows, num_cols, num_cols*idx+6)
        if idx is 0:
            plt.title("12.5m transformed")

        h = torch.rfft(h, 2, normalized=True)
        h = h.pow(2).sum(-1).sqrt()
        plt.imshow(h.squeeze().detach().cpu(),
                   interpolation='none')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("images/gan_samples_{}.png".format(epoch))
    plt.close()


def main():
    device = torch.device(args.device)
    #determining the input data parameters
    if args.is_fk_data:
        output_dim = [127,62]
        filename_x='data_fk_25'
        filename_y='data_fk_125'
    else:
        output_dim = [251,121]
        filename_x='data_25'
        filename_y='data_125'

    #getting the data set and standartising it
    dataset = Data(filename_x = filename_x, filename_y = filename_y, transforms=transforms.Compose([
        transforms.ToTensor()
        #this is the actual statistic, not the 0,1
        #transforms.Normalize(torch.tensor(-4.4713e-07).float(), torch.tensor(0.1018).float())
    ]))
    #train test split
    test_size = round(len(dataset)*args.test_percentage)
    train_size = len(dataset) - test_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    #dataloaders
    train_dataloader = DataLoader(train_data, batch_size = args.batch_size, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size = 4)

    # Init generator model.
    if args.model == "SRCNN":
        generator = SRCNN().to(device)
    elif args.model == "EDSR":
        generator = EDSR(n_resblocks = args.num_res_blocks, output_dim = output_dim, latent_dim=args.latent_dim).to(device)

    # Init discriminator model.
    discriminator = Discriminator().to(device)

    #optimisers
    optim_G = optim.Adam(generator.parameters(), lr = args.lr)
    optim_D = optim.Adam(discriminator.parameters(), lr = args.lr)




    plot_G = []
    plot_D = []

    for epoch in range(args.n_epochs):
        loss_G, loss_D, mean_psnr = train_epoch(generator, discriminator, optim_G, optim_D, train_dataloader, device)

        # Report model performance.
        print(f"Epoch: {epoch}, G: {loss_G}, D: {loss_D}, PSNR: {mean_psnr}")
        if epoch%args.save_interval == 0:
            plot_samples(generator, test_dataloader, epoch)

        plot_D.append(loss_D)
        plot_G.append(loss_G)

    plt.figure()
    plt.plot(plot_D, label="Discriminator loss")
    plt.plot(plot_G, label="Generator loss")
    plt.legend()
    plt.savefig("images/gan_loss.png")
    plt.close()

if  __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='dimensionality of the latent space, only relevant for EDSR')
    parser.add_argument('--num_res_blocks', type=int, default=16,
                        help='Number of resblocks in model, only relevant for EDSR')
    parser.add_argument('--model', type=str, default="EDSR",
                        help="Model type. EDSR or SRCNN")
    parser.add_argument('--is_fk_data', type=bool, default=False,
                        help='Is fourier data')
    parser.add_argument('--test_percentage', type=float, default=0.1,
                        help='size of the test set')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='save every SAVE_INTERVAL epochs')
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")

    args = parser.parse_args()

    main()