import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal

from matplotlib.pyplot import imread
import numpy as np
import random
from pathlib import Path

DATA_FOLDER = Path(".")

def sample_gaussian(mean, cov, n_samples):
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean).float()
    if mean.ndim == 0:
        mean = mean.unsqueeze(0)
        
    if not isinstance(cov, torch.Tensor):
        cov = torch.tensor(cov).float()
    if cov.ndim == 0:
        cov = cov * torch.eye(len(mean))
    elif cov.ndim == 1:
        cov = torch.diag(cov)

    if not isinstance(n_samples, tuple):
        n_samples = (n_samples,)
    
    distri = MultivariateNormal(mean, cov)
    samples = distri.sample(n_samples)
    return samples


def sample_mixture_of_gaussians(means, stds, n_samples):
    samples = torch.cat([sample_gaussian(m, s, n) for m, s, n in zip(means, stds, n_samples)])
    samples = samples[torch.randperm(len(samples))]
    return samples

def load_image(fname):
    img = np.mean(imread(fname), axis=2)  # Grayscale
    img = (img[:, :]) / 255.0
    return 1 - img  # black = 1, white = 0

def cloud_transform(image, n_samples=1000):
    image = image > 0.1
    samples = np.vstack([np.where(image)[1] / image.shape[1],
                         1 - np.where(image)[0] / image.shape[0]]).T
    samples = samples - samples.mean(axis=0)
    samples = samples * 2
    # add a little bit of noise to avoid the sampling patterns
    samples += np.random.randn(samples.shape[0], 2) * 2e-3
    samples = samples[random.sample([i for i in range(len(samples))], n_samples)]
    return samples

class DataLoaders(): #TODO: simplify
    def __init__(self, dataset_name, batch_size_train, batch_size_test):
        self.dataset_name = dataset_name
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test

    def load_data(self):

        if self.dataset_name == "gaussians":
            n_train = 1000
            n_val   = 1000
            n_test  = 1000

            means = (-2., 2.)
            vars  = (1/4, 1/4)
            train_x = sample_mixture_of_gaussians(means, vars, (n_train//2, n_train - n_train//2))
            val_x   = sample_mixture_of_gaussians(means, vars, (n_val  //2, n_val   - n_val  //2))
            test_x  = sample_mixture_of_gaussians(means, vars, (n_test //2, n_test  - n_test //2))

            train_dataset = TensorDataset(train_x, torch.zeros(len(train_x)))
            val_dataset   = TensorDataset(val_x  , torch.zeros(len(val_x  )))
            test_dataset  = TensorDataset(test_x , torch.zeros(len(test_x )))

            train_loader = DataLoader(train_dataset, batch_size = self.batch_size_train, shuffle=True)
            val_loader   = DataLoader(val_dataset  , batch_size = self.batch_size_test , shuffle=True)
            test_loader  = DataLoader(test_dataset , batch_size = self.batch_size_test , shuffle=True)
        
        elif self.dataset_name == "gaussians2D":
            n_train = 1000
            n_val   = 1000
            n_test  = 1000

            means = ((-1, 0), (1, 0))
            vars  = ((1/4, 1/16), (1/16, 1/4))
            train_x = sample_mixture_of_gaussians(means, vars, (n_train//2, n_train - n_train//2))
            val_x   = sample_mixture_of_gaussians(means, vars, (n_val  //2, n_val   - n_val  //2))
            test_x  = sample_mixture_of_gaussians(means, vars, (n_test //2, n_test  - n_test //2))

            train_dataset = TensorDataset(train_x, torch.zeros(len(train_x)))
            val_dataset   = TensorDataset(val_x  , torch.zeros(len(val_x  )))
            test_dataset  = TensorDataset(test_x , torch.zeros(len(test_x )))

            train_loader = DataLoader(train_dataset, batch_size = self.batch_size_train, shuffle=True)
            val_loader   = DataLoader(val_dataset  , batch_size = self.batch_size_test , shuffle=True)
            test_loader  = DataLoader(test_dataset , batch_size = self.batch_size_test , shuffle=True)

        elif self.dataset_name == "obelix":
            train_x = torch.tensor(cloud_transform(load_image(DATA_FOLDER / "obelix.jpg"), n_samples=5000)).float()
            val_x   = torch.tensor(cloud_transform(load_image(DATA_FOLDER / "obelix.jpg"), n_samples=5000)).float()
            test_x  = torch.tensor(cloud_transform(load_image(DATA_FOLDER / "obelix.jpg"), n_samples=5000)).float()

            train_dataset = TensorDataset(train_x, torch.zeros(len(train_x)))
            val_dataset   = TensorDataset(val_x  , torch.zeros(len(val_x  )))
            test_dataset  = TensorDataset(test_x , torch.zeros(len(test_x )))

            train_loader = DataLoader(train_dataset, batch_size = self.batch_size_train, shuffle=True)
            val_loader   = DataLoader(val_dataset  , batch_size = self.batch_size_test , shuffle=True)
            test_loader  = DataLoader(test_dataset , batch_size = self.batch_size_test , shuffle=True)
            
        data_loaders = {'train': train_loader,
                        'test': test_loader, 'val': val_loader}

        return data_loaders
    


