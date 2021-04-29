# Bayesian Generative Adversarial Networks

Pytorch implementation of the [Bayesian GAN](https://arxiv.org/abs/1705.09558) presentend by Wilson et al. 



## Prerequisite
- Pytorch
- Numpy
- Scilkit-learn
- Scipy

## Run
if you want to replicat the results, download the repo and run the script as follow:


```cd IFT6756-BayesianGAN/src/
```python bgan.py dataset prior optimizer bool

- dataset = {mnist, celeba}
- prior = {normal, uniform, cauchy, laplace}
- optimizer = {adam, SGLD, pSGLD, SGHMC}
- bool = {0, 1}. Set bool to 0 if you want to run a standard DCGAN

The code in this repo is based on the following repos/links:

- 
