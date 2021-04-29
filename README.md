# Bayesian Generative Adversarial Networks

Pytorch implementation of the [Bayesian GAN](https://arxiv.org/abs/1705.09558) presentend by Wilson et al. 



## Prerequisite
- Pytorch
- Numpy
- Scilkit-learn
- Scipy

## Run
To replicate the results, clone the repo and run the script as follows:


```
git clone https://github.com/emiled16/IFT6756-BayesianGAN.git
cd IFT6756-BayesianGAN/src/
python bgan.py dataset prior optimizer bool
```


- dataset = {mnist, celeba}
- prior = {normal, uniform, cauchy, laplace}
- optimizer = {adam, SGLD, pSGLD, SGHMC}
- bool = {0, 1}. Set bool to 0 if you want to run a standard DCGAN

The code in this repo is based on the following repos:

- [DCGAN TUTORIAL](https://github.com/pytorch/tutorials/blob/master/beginner_source/dcgan_faces_tutorial.py)
- [mltrain-nips-2017](https://github.com/vasiloglou/mltrain-nips-2017/tree/master/ben_athiwaratkun/pytorch-bayesgan)
- [Fréchet Inception Distance (FID) for Pytorch](https://github.com/hukkelas/pytorch-frechet-inception-distance)
- [FID](https://github.com/bioinf-jku/TTUR/blob/master/fid.py)
- [Inception Score Pytorch](https://github.com/sbarratt/inception-score-pytorch)
