import sys
import argparse
import os
import numpy as np
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch
from torch import nn
from torchvision.models import inception_v3
import cv2
import multiprocessing
import numpy as np
import glob
import os
from scipy import linalg
from torchvision.models.inception import inception_v3
from scipy.stats import entropy
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from download_dataset import download_CELEBA, download_MNIST
from Opt import *


argument_list = sys.argv[1:]
DATASET = argument_list[0]
prior_dist = argument_list[1]
optim_method = argument_list[2]
if argument_list[3] == '1':
    bayes = True
else:
    bayes = False

n_epochs = 30
batch_size = 64
lr = 0.00002
Jg = 1
Jd = 1
M = 1
latent_dim = 100
###############################################################
if bayes == True:
    dir = './{}/{}/{}'.format(DATASET, optim_method, prior_dist)
else:
    dir = './{}/{}/standard'.format(DATASET, optim_method)
noise_SGD_std = 0.01
noise_SGD = torch.distributions.Normal(0, 2*lr*noise_SGD_std)
###############################################################
sample_interval = 400
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
inception_model = inception_v3(pretrained=True, transform_input=False).cuda()
up_mnist = nn.Upsample(size=(64, 64), mode='bilinear')


def save_model(epoch):
    for i, G in enumerate(gen_list):
        torch.save(G.state_dict(), dir + "/models/genrator_{}_e{}".format(i, epoch))
    for i, D in enumerate(disc_list):
        torch.save(G.state_dict(), dir + "/models/discriminator_{}_e{}".format(i, epoch))


def to_cuda(elements):
    """
    Transfers elements to cuda if GPU is available
    Args:
        elements: torch.tensor or torch.nn.module
        --
    Returns:
        elements: same as input on GPU memory, if available
    """
    if torch.cuda.is_available():
        return elements.cuda()
    return elements


def adapt_mnist(imgs):
    if DATASET == 'mnist':
        imgs = up_mnist(imgs)
        imgs = imgs.repeat(1, 3, 1, 1)
        return imgs
    return imgs


class PartialInceptionNetwork(nn.Module):

    def __init__(self, transform_input=True):
        super().__init__()
        self.inception_network = inception_v3(pretrained=True)
        self.inception_network.Mixed_7c.register_forward_hook(self.output_hook)
        self.transform_input = transform_input

    def output_hook(self, module, input, output):
        self.mixed_7c_output = output

    def forward(self, x):
        """
        Args:
            x: shape (N, 3, 299, 299) dtype: torch.float32 in range 0-1
        Returns:
            inception activations: torch.tensor, shape: (N, 2048), dtype: torch.float32
        """
        assert x.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" +\
                                             ", but got {}".format(x.shape)
        x = x * 2 - 1
        # Trigger output hook
        self.inception_network(x)
        activations = self.mixed_7c_output
        activations = torch.nn.functional.adaptive_avg_pool2d(activations, (1, 1))
        activations = activations.view(x.shape[0], 2048)
        return activations


def get_activations(images, batch_size):
    """
    Calculates activations for last pool layer for all iamges
    --
        Images: torch.array shape: (N, 3, 299, 299), dtype: torch.float32
        batch size: batch size used for inception network
    --
    Returns: np array shape: (N, 2048), dtype: np.float32
    """
    assert images.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" +\
                                              ", but got {}".format(images.shape)

    num_images = images.shape[0]
   
    inception_network.eval()
    n_batches = int(np.ceil(num_images / batch_size))
    inception_activations = np.zeros((num_images, 2048), dtype=np.float32)
    for batch_idx in range(n_batches):
        start_idx = batch_size * batch_idx
        end_idx = batch_size * (batch_idx + 1)

        ims = images[start_idx:end_idx]
        ims = to_cuda(ims)
        activations = inception_network(ims)
        activations = activations.detach().cpu().numpy()
        assert activations.shape == (ims.shape[0], 2048), "Expected output shape to be: {}, but was: {}".format((ims.shape[0], 2048), activations.shape)
        inception_activations[start_idx:end_idx, :] = activations
    return inception_activations


def calculate_activation_statistics(images, batch_size):
    """Calculates the statistics used by FID
    Args:
        images: torch.tensor, shape: (N, 3, H, W), dtype: torch.float32 in range 0 - 1
        batch_size: batch size to use to calculate inception scores
    Returns:
        mu:     mean over all activations from the last pool layer of the inception model
        sigma:  covariance matrix over all activations from the last pool layer 
                of the inception model.
    """
    act = get_activations(images, batch_size)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


# Modified from: https://github.com/bioinf-jku/TTUR/blob/master/fid.py
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_fid(images1, images2, use_multiprocessing, batch_size):
    """ Calculate FID between images1 and images2
    Args:
        images1: np.array, shape: (N, H, W, 3), dtype: np.float32 between 0-1 or np.uint8
        images2: np.array, shape: (N, H, W, 3), dtype: np.float32 between 0-1 or np.uint8
        use_multiprocessing: If multiprocessing should be used to pre-process the images
        batch size: batch size used for inception network
    Returns:
        FID (scalar)
    """
    mu1, sigma1 = calculate_activation_statistics(images1, batch_size)
    mu2, sigma2 = calculate_activation_statistics(images2, batch_size)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


def IS(FAKE_img):
    inception_model.eval()

    def get_pred(x):
        # x = x.repeat(1, 3, 1, 1)
        # x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()  
    preds = np.zeros((N, 1000))
    batchv = Variable(FAKE_img)
    preds = get_pred(batchv)
    py = np.mean(preds, axis=0)
    n = preds.shape[0]
    splits = 10
    split_scores = []
    for k in range(splits):
        part = preds[k * (n // splits): (k+1) * (n // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))
    return np.mean(split_scores), np.std(split_scores)


def eval_discriminator():
    z = Variable(Tensor(np.random.normal(0, 1, (batch_size, latent_dim, 1, 1))))
    # getting fake images
    fakes_img = []
    for G in gen_list:
        fakes_img.append(G(z))
    fake_img = torch.cat(fakes_img)
    fake_label = Variable(Tensor(batch_size*Jg*M, 1).fill_(0.0), requires_grad=False)
    # getting real images
    imgs = []
    for i in range(Jg*M):
        imgs.append(next(iter(testloader))[0])
    img = torch.cat(imgs)
    img = adapt_mnist(img)
    print(img.shape)
    true_label = Variable(Tensor(batch_size*Jg*M, 1).fill_(1.0), requires_grad=False)

    preds = []
    real = []
    for D in disc_list:
        preds.append(D(fake_img).detach().cpu().numpy().ravel())
        real.append(fake_label.detach().cpu().numpy().ravel())
        preds.append(D(img.cuda()).detach().cpu().numpy().ravel())
        real.append(true_label.detach().cpu().numpy().ravel())

    preds = np.concatenate(preds, axis=None).ravel()
    real = np.concatenate(real, axis=None).ravel()

    f1 = f1_score(real, preds > 0.5)
    precision = precision_score(real, preds > 0.5)
    recall = recall_score(real, preds > 0.5)
    return f1, precision, recall


def eval():
    up = nn.Upsample(size=(299, 299), mode='bilinear')
    # creating noise for evaluation
    z = Variable(Tensor(np.random.normal(0, 1, (batch_size, latent_dim, 1, 1))))
    # getting fake images
    fakes_img = []
    for G in gen_list:
        fakes_img.append(G(z))
    fake_img = torch.cat(fakes_img)
    # getting real images
    imgs = []
    for i in range(Jg*M):
        imgs.append(next(iter(testloader))[0])
    img = torch.cat(imgs)
    # reshaping
    if DATASET == 'mnist':
        real_img = img.repeat(1, 3, 1, 1)
        real_img = up(real_img)
        #FAKE_img = fake_img.repeat(1, 3, 1, 1)
        FAKE_img = up(fake_img)
    else:
        FAKE_img = up(fake_img)
        real_img = up(img)
    # computing IS
    IS_mu, IS_std = IS(FAKE_img)
    # Computing FID
    FID = calculate_fid(FAKE_img, real_img, False, 6)
    return IS_mu, IS_std, FID





nc = 3
ngf = 64


class Generator(nn.Module):
    def __init__(self, ngpu=1):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


ndf = 64


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(

            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# weight initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def select_optim(method='adam'):

    ###########################################################################
    def SGLD(model):
        return torch.optim.SGD(model.parameters(), lr=0.00002, momentum=0)

    def SGHMC(model):
        return torch.optim.SGD(model.parameters(), lr=0.00002, momentum=0.7)

    def adam_HMC(model):
        return torch.optim.Adam(model.parameters(), lr=0.00002, betas=(0.5, 0.999))

    def pSGLD(model):
        return torch.optim.RMSprop(model.parameters(), lr=0.0001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    ###########################################################################
    if method == 'adam':
        return adam_HMC

    if method == 'SGLD':
        return SGLD

    if method == 'pSGLD':
        return pSGLD

    if method == 'SGHMC':
        return SGHMC


def noise_loss(params):
    noise_loss = 0
    for var in params:
        noise_loss += (noise_SGD.sample(var.shape).cuda()*var).sum()
    return noise_loss / N


# DOWNLOAD THE DATA

if DATASET == 'mnist':
    img_size = 28
    channels = 1
    img_shape = (channels, img_size, img_size)
    N = 50000
    dataloader, testloader = download_MNIST(64)
if DATASET == 'celeba':
    img_size = 64
    channels = 3
    img_shape = (channels, img_size, img_size)
    N = 50000
    dataloader, testloader = download_CELEBA(64)


inception_network = PartialInceptionNetwork()
inception_network = to_cuda(inception_network)


noise_SGD = torch.distributions.Normal(0, 2*lr*noise_SGD_std)

# others priors
# uniform parameters:
low = torch.tensor(-10.).cuda()
high = torch.tensor(10.).cuda()
# Gamma parameters:
concentration = 1.
rate = 1.
# Cauchy parameters:
loc_c = 0.
scale_c = 1.
# half-Cauchy parameters:
scale_hc = 1.
# Laplace parameters:
loc_l = 0.
scale_l = 1.
if prior_dist == 'normal':
    prior = torch.distributions.Normal(0, 1)

if prior_dist == 'uniform':
    prior = torch.distributions.uniform.Uniform(low, high)

if prior_dist == 'gamma':
    prior = torch.distributions.gamma.Gamma(concentration, rate)

if prior_dist == 'cauchy':
    prior = torch.distributions.cauchy.Cauchy(loc_c, scale_c)

if prior_dist == 'half_cauchy':
    prior =  torch.distributions.half_cauchy.HalfCauchy(scale_hc)

if prior_dist == 'laplace':
    prior = torch.distributions.laplace.Laplace(loc_l, scale_l)


def prior_loss(params, prior=prior):
    prior_loss = 0
    for var in params:
        prior_loss += torch.mean(prior.log_prob(var)).cuda()
    return prior_loss / N


adversarial_loss = torch.nn.BCELoss()
init_optim = select_optim(optim_method)

gen_list = []
gen_optim_list = []
for idx in range(Jg):
    for idxx in range(M):
        G = Generator().cuda()
        G.apply(weights_init)
        optimizer_G = init_optim(G)
        gen_list.append(G)
        gen_optim_list.append(optimizer_G)


disc_list = []
disc_optim_list = []
for idx in range(Jd):
    for idxx in range(M):
        D = Discriminator().cuda()
        D.apply(weights_init)
        optimizer_D = init_optim(D)
        disc_list.append(D)
        disc_optim_list.append(optimizer_D)

os.makedirs(dir + '/images', exist_ok=True)
os.makedirs(dir + '/models', exist_ok=True)


train_iter = 0
for epoch in range(n_epochs):
    G_losses = 0
    D_losses = 0
    D_preds = []
    D_true = []
    train_iter = 0
    for i, (imgs, _) in enumerate(dataloader):

        train_iter += 1

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1, 1, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1, 1, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        for idx_d in range(Jd):
            for idxx_d in range(M):
                id_D = idx_d*M + idxx_d
                discriminator = disc_list[id_D]
                optim_D = disc_optim_list[id_D]
                optim_D.zero_grad()

                # -----------------
                #  Train Generator
                # -----------------
                fakes_img = []
                for optim_G in gen_optim_list:
                    optim_G.zero_grad()

                for idx_g in range(Jg):

                    # Sample noise as generator input
                    z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim, 1, 1))))

                    for idxx_g in range(M):
                        optim_D.zero_grad()
                        # Generate a batch of images
                        idG = idx_g*M + idxx_g
                        generator = gen_list[idG]
                        optim_G = gen_optim_list[idG]
                        gen_imgs = generator(z)
                        fakes_img.append(gen_imgs)

                        # Loss measures generator's ability to fool the discriminator
                        g_loss = adversarial_loss(discriminator(gen_imgs), valid).cuda()

                        if bayes == True:
                            g_loss += prior_loss(generator.parameters())
                            g_loss += noise_loss(generator.parameters())

                        g_loss.backward()
                        optim_G.step()
                        G_losses += g_loss.cpu().detach()

                # ---------------------
                #  Train Discriminator
                # ---------------------
                fake_img = torch.cat(fakes_img)
                optim_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(discriminator(adapt_mnist(real_imgs)), valid)
                fake_loss = adversarial_loss(discriminator(fake_img.detach()), torch.cat(Jg*M*[fake]))  # keep in mind to change that !!!!!!!!!!!!!!!!!!!!!
                d_loss = (real_loss*Jg*M + fake_loss) / 2
                if bayes == True:
                    d_loss += prior_loss(discriminator.parameters())
                    d_loss += noise_loss(discriminator.parameters())
                D_losses += d_loss.cpu().detach()

                d_loss.backward()
                optim_D.step()

    ############################################################
    IS_mu, IS_std, FID = eval()
    f1, precision, recall = eval_discriminator()
    print(
            "[Epoch %d/%d] [D loss: %f] [G loss: %f][IS:  %f][FID: %f][F1: %f][Precision: %f][Recall: %f]"
            % (epoch, n_epochs, D_losses/train_iter, G_losses/train_iter, IS_mu,
               FID, f1, precision, recall)
        )

    # if batches_done % sample_interval == 0:
    save_image(gen_imgs.data[:25], dir + "/images/%d.png" % epoch, nrow=5, normalize=True)
    save_image(fake_img.data, dir + "/images/full%d.png" % epoch, nrow=5, normalize=True)
    save_model(epoch)
