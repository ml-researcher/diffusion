import torch
from torchvision.utils import save_image
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

def sampling(e, model):
    model.eval()
    with torch.no_grad():
        sampled_images = diffusion.sample(batch_size = 64)
        save_image(sampled_images, './samples/sample_{}_eval'.format(e) + '.png')

if __name__ == '__main__':
    epoch = 49
    diffusion = torch.load('models/epoch{}.pt'.format(epoch))
    # You may change variance to produce different samples.
    diffusion.sigma = diffusion.betas*diffusion.sqrt_rev_alphas*diffusion.sqrt_rev_one_minus_alphas_cumprod*1.5
    sampling(epoch, diffusion)