import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from aae import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

#import albumentations as A
#from albumentations.pytorch import ToTensorV2


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--exp_name", type=str, default="vessel", help="name of the experiment")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--nf", type=int, default=32, help="number of filters")
parser.add_argument("--lr", type=float, default=0.002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=16, help="dimensionality of the latent code")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model checkpoints")
parser.add_argument("--sample_interval", type=int, default=400, help="interval saving generator samples")
opt = parser.parse_args()

cuda = torch.cuda.is_available()

# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.exp_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.exp_name, exist_ok=True)


# Use binary cross-entropy loss
adversarial_loss = torch.nn.BCELoss()
pixelwise_loss = torch.nn.L1Loss()

# Initialize generator and discriminator
encoder = Encoder(in_ch=opt.channels, nf=opt.nf, latent_dim=opt.latent_dim)
decoder = Decoder(out_ch=opt.channels, nf=opt.nf, latent_dim=opt.latent_dim)
discriminator = Discriminator(latent_dim=opt.latent_dim)

if cuda:
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    pixelwise_loss.cuda()


# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

transforms_A = transforms.Compose([
   transforms.Resize(128),
   transforms.ToTensor(),
   transforms.Normalize(
       mean=[0.5, 0.5, 0.5],
       std=[0.5, 0.5, 0.5]
   )
])


dataloader = DataLoader(
    ImageDataset("../../data/unet_segmentations_binary/", "train",  transforms_A),
    batch_size=opt.batch_size,
    shuffle=True,
)

val_dataloader = DataLoader(
    ImageDataset("../../data/unet_segmentations_binary/", "val",  transforms_A),
    batch_size=50,
    shuffle=True,
)

def sample_image(n_row, batches_done):
    """Saves a grid of generated digits"""
    # Sample noise
    z = torch.randn(n_row ** 2, opt.latent_dim).type(Tensor)
    gen_imgs = decoder(z)
    save_image(gen_imgs.data, "images/%s/%s.png" % (opt.exp_name, batches_done), nrow=n_row, normalize=True)


# ----------
#  Training
# ----------
for epoch in range(opt.n_epochs):
    for i, imgs in enumerate(dataloader):

        # Adversarial ground truths
        valid = torch.ones(imgs.shape[0], 1).type(Tensor)
        fake = torch.zeros(imgs.shape[0], 1).type(Tensor)

        # Configure input
        real_imgs = imgs.type(Tensor)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        encoded_imgs = encoder(real_imgs)
        decoded_imgs = decoder(encoded_imgs)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(encoded_imgs), valid) + 100 * pixelwise_loss(
            decoded_imgs, real_imgs
        )

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as discriminator ground truth
        z = torch.randn(imgs.shape[0], opt.latent_dim).type(Tensor)

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(z), valid)
        fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )
        
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=5, batches_done=batches_done)