import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from torchvision.utils import save_image
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

batch_size = 100
sample_freq = 5
epoch = 50
dim_z = 10

train_dataset = datasets.MNIST(root='./mnist', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist', train=False, transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

def train_one_epoch(e, model, optimizer):
    model.train()
    pbar = tqdm(train_loader)
    for data, _ in pbar:
        if torch.cuda.is_available():
            data = data.cuda()
        optimizer.zero_grad()
        loss = model(data)
        loss.backward()
        optimizer.step()
        pbar.set_description("epoch {} train loss: {:.2f}".format(e, loss.item()))

def sampling(e, model):
    model.eval()
    with torch.no_grad():
        sampled_images = diffusion.sample(batch_size = 64)
        save_image(sampled_images, './samples/sample_{}'.format(e) + '.png')

if __name__ == '__main__':
    model = Unet(
        channels = 1,
        dim = 64,
        dim_mults = (1, 2, 4)
    )
    diffusion = GaussianDiffusion(
        model,
        image_size = 28,
        timesteps = 1000,   # number of steps
        loss_type = 'l2'    # L1 or L2
    )
    if torch.cuda.is_available():
        diffusion = diffusion.cuda()
    optimizer = optim.Adam(diffusion.parameters())
    os.makedirs('./samples', exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    for e in range(epoch):
        train_one_epoch(e, diffusion, optimizer)
        sampling(e, diffusion)
        torch.save(diffusion, 'models/epoch{}.pt'.format(e))