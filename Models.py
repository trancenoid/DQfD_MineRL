import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
from torch import optim
import cv2
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchvision.io import read_image
import os
import pickle

def make_vae():
	class Args:
  		pass

	args = Args()

	args.batch_size = 256
	args.img_size = 64
	args.epochs = 10
	args.lr = 1e-3
	args.zdim = 64
	args.cuda = True
	args.recl_weight = 1.0
	args.log_interval = 50
	args.model = 'sigma_vae'  # Which model to use:  mse_vae,  gaussian_vae, or sigma_vae or optimal_sigma_vae
	args.dream_inp_len = 4

	if not torch.cuda.is_available():
	  args.cuda = False
	device = torch.device("cuda" if args.cuda else "cpu")

	print("Loading VAE Model...", end = '')

	with open("../MineRL_PlanningAgent/KMM_SS_64_1.7K_.pkl", 'rb') as f:
	    km = pickle.load(f)

	def smooth_obs(image):

	    colors = image.reshape(64*64,3)
	    colors = km.cluster_centers_[km.predict(colors)]
	    colors = colors.reshape(64,64,3).astype(int)
	    image = torch.tensor(colors.transpose(2,0,1)).to(dtype = torch.float32)/255.0
	    return image



	vae_model = ConvVAE(device, z_dim = args.zdim, img_channels = 3, args = args).to(device)
	checkpoint = torch.load('../MineRL_PlanningAgent/checkpoints/SmootherVAE_261021_64x64_64L_4', 'cuda')
	vae_model.load_state_dict(checkpoint)
	vae_model.eval()
	print("done")

	def get_obs_enc(obs):

#         obs = obs.transpose(-1,0,1).copy()
	    img = smooth_obs(obs)
	    _ , mu , _ = vae_model(img.unsqueeze(0).to(device))
	    mu = mu.detach().cpu().numpy()[0]
	    return mu

	return vae_model,get_obs_enc

def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, n_channels):
        super(UnFlatten, self).__init__()
        self.n_channels = n_channels
    
    def forward(self, input):
        size = int((input.size(1) // self.n_channels) ** 0.5)
        return input.view(input.size(0), self.n_channels, size, size)


class ConvVAE(nn.Module):
    def __init__(self, device='cuda', z_dim=2, img_channels=3, args=None):
        super().__init__()
        self.batch_size = args.batch_size
        self.device = device
        self.z_dim = z_dim
        self.img_channels = img_channels
        self.model = args.model
        self.img_size = args.img_size
        filters_m = 32

        ## Build network
        self.encoder = self.get_encoder(self.img_channels, filters_m)

        # output size depends on input image size, compute the output size
        demo_input = torch.ones([1, self.img_channels, self.img_size, self.img_size])
        h_dim = self.encoder(demo_input).shape[1]
        
        # map to latent z
        self.fc11 = nn.Linear(h_dim, self.z_dim)
        self.fc12 = nn.Linear(h_dim, self.z_dim)

        # decoder
        self.fc2 = nn.Linear(self.z_dim, h_dim)
        self.decoder = self.get_decoder(filters_m, self.img_channels)
        
        self.log_sigma = 0 
        if self.model == 'sigma_vae':
            ## Sigma VAE
            self.log_sigma = torch.nn.Parameter(torch.full((1,), 0.0)[0], requires_grad=args.model == 'sigma_vae')
        

    @staticmethod
    def get_encoder(img_channels, filters_m):
        return nn.Sequential(
            nn.Conv2d(img_channels, filters_m, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters_m, 2 * filters_m, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * filters_m, 4 * filters_m, (5, 5), stride=2, padding=2),
            nn.ReLU(),
            Flatten()
        )
    
    @staticmethod
    def gaussian_nll(mu, log_sigma, x):
        return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)

    @staticmethod
    def get_decoder(filters_m, out_channels):
        return nn.Sequential(
            UnFlatten(4 * filters_m),
            nn.ConvTranspose2d(4 * filters_m, 2 * filters_m, (6, 6), stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * filters_m, filters_m, (6, 6), stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(filters_m, out_channels, (5, 5), stride=1, padding=2),
            nn.Sigmoid(),
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc11(h), self.fc12(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(self.fc2(z))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def sample(self, n):
        sample = torch.randn(n, self.z_dim).to(self.device)
        return self.decode(sample)

    def reconstruction_loss(self, x_hat, x):
        """ Computes the likelihood of the data given the latent variable,
        in this case using a Gaussian distribution with mean predicted by the neural network and variance = 1 """
        
        if self.model == 'gaussian_vae':
            # Naive gaussian VAE uses a constant variance
            log_sigma = torch.zeros([], device=x_hat.device)
        elif self.model == 'sigma_vae':
            # Sigma VAE learns the variance of the decoder as another parameter
            log_sigma = self.log_sigma
        elif self.model == 'optimal_sigma_vae':
            log_sigma = ((x - x_hat) ** 2).mean([0,1,2,3], keepdim=True).sqrt().log()
            self.log_sigma = log_sigma.item()
        else:
            raise NotImplementedError

        # Learning the variance can become unstable in some cases. Softly limiting log_sigma to a minimum of -6
        # ensures stable training.
        log_sigma = softclip(log_sigma, -6)
        
        
        rec = self.gaussian_nll(x_hat, log_sigma, x).sum()
    
        return rec

    def loss_function(self, recon_x, x, mu, logvar):
        # Important: both reconstruction and KL divergence loss have to be summed over all element!
        # Here we also sum the over batch and divide by the number of elements in the data later
        if self.model == 'mse_vae':
            rec = torch.nn.MSELoss()(recon_x, x)
        else:
            rec = self.reconstruction_loss(recon_x, x)
        
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return rec, kl