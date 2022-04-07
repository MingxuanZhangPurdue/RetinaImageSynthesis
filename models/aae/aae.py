import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def reparameterization(mu, logvar):
    std = torch.exp(logvar/2)
    sampled_z = torch.randn(mu.shape).type(Tensor)
    z = sampled_z*std + mu
    return z


# 5,4,2 -> divide by 4
# 3,2,1 -> divide by 2
# 2,1,0 -> minus 1



# Takes 1x128x128 images
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=2, p=1, dropout_p=0):
        
        super(ConvBlock, self).__init__()
        
        if dropout_p == 0:
            self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_p)
            )
            
    def forward(self, x):
        return self.convblock(x)
    
    
class ConvTransposeBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=2, p=1, op=1, dropout_p=0):
        
        super(ConvTransposeBlock, self).__init__()
        
        if dropout_p == 0:
            self.convtransposeblock = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, k, s, p, op),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.convtransposeblock = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, k, s, p, op),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_p)
            )
            
    def forward(self, x):
        return self.convtransposeblock(x)
    

class UpsampleConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, dropout_p=0):
        
        super(UpsampleConvBlock, self).__init__()
        
        if dropout_p == 0:
            self.convtransposeblock = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, k, s, p),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.convtransposeblock = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, k, s, p),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_p),
            )
            
    def forward(self, x):
        return self.convtransposeblock(x)            
        
class Encoder(nn.Module):
    def __init__(self, in_ch, nf, latent_dim):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            ConvBlock(in_ch, nf),
            
            ConvBlock(nf, nf*2),
            
            ConvBlock(nf*2, nf*4),
            
            ConvBlock(nf*4, nf*8),
            
            ConvBlock(nf*8, nf*8),
            
            ConvBlock(nf*8, nf*8),
            
            ConvBlock(nf*8, nf*8, k=2, s=1, p=0),
            
            nn.Flatten()
        )

        self.mu = nn.Linear(nf*8, latent_dim)
        self.logvar = nn.Linear(nf*8, latent_dim)

    def forward(self, img):
        x = self.model(img)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z

    
# 3,2,1,1 -> scale by 2

class Decoder(nn.Module):
    def __init__(self, out_ch, nf, latent_dim):
        super(Decoder, self).__init__()
        
        self.linear = nn.Linear(latent_dim, nf*8)
        self.nf = nf
        
        self.model = nn.Sequential(
            
            ConvTransposeBlock(nf*8, nf*8), #2x2
            
            ConvTransposeBlock(nf*8, nf*8), #4x4
            
            ConvTransposeBlock(nf*8, nf*8), #8x8
            
            ConvTransposeBlock(nf*8, nf*8), #16x16
            
            ConvTransposeBlock(nf*8, nf*4), #32x32,
            
            ConvTransposeBlock(nf*4, nf*2), #64x64
            
            ConvTransposeBlock(nf*2, nf), #128x128
            
            nn.Conv2d(nf, out_ch, kernel_size=1, stride=1, padding=0),
            
            nn.Sigmoid(),
        )
        
        
        
        """
        self.model = nn.Sequential(
            UpsampleConvBlock(nf*8, nf*8), #2x2
            
            UpsampleConvBlock(nf*8, nf*8, dropout_p=0.5), #4x4
            
            UpsampleConvBlock(nf*8, nf*8, dropout_p=0.5), #8x8
            
            UpsampleConvBlock(nf*8, nf*8, dropout_p=0.5), #16x16
            
            UpsampleConvBlock(nf*8, nf*4), #32x32
            ConvBlock(nf*4, nf*4, s=1),
            
            UpsampleConvBlock(nf*4, nf*2), #64x64
            ConvBlock(nf*2, nf*2, s=1),
            
            UpsampleConvBlock(nf*2, nf), #128x128
            ConvBlock(nf, nf, s=1),
            
            nn.Conv2d(nf, out_ch, kernel_size=1, stride=1, padding=0),
            
            nn.Sigmoid(),
        )
        """

    def forward(self, z):
        
        x = self.linear(z)
        x = x.view(-1, self.nf*8, 1, 1)
        img = self.model(x)
        return img


class Discriminator(nn.Module):
    def __init__(self, latent_dim, nf):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nf, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity