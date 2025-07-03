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
    """Sample a multivariate gaussian.

    Args:
        mean (Tensor|ndarray|list|tuple|float): Mean of the gaussian distribution.
        cov (Tensor|ndarray|list|tuple|float): Covariance matrix of the distribution. If a float s is provided, \
            then the covariance matrix is assumed to be s*torch.eye(d). If a 1d tensor vars is provided, then the \
            covariance matrix is assumed to be torch.diag(vars).
        n_samples (int): number of samples

    Returns:
        Tensor: shape (n_samples, d).
    """
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


def sample_mixture_of_gaussians(means, covars, n_samples):
    """Sample a mixture of gaussians.

    Args:
        means (tuple): each element represent the mode of one of the gaussian distributions considered 
            in the mixture (can be a float for 1D distributions, a tensor, an array, a tuple, a list, ...)
        covars (tuple): each element represents the covariance of one of the gaussian distributions considered
            in the mixture (can be float for isotropic gaussians, a tensor, an array, a tuple, a list ->
            if 1D, it is considered as the diagonal values of the covariance matrix; if 2D, must be a SPD matrix)
        n_samples (tuple): number of samples for each of the gaussian distributions.

    Returns:
        Tensor: (sum(n_samples), d) where d is infered from means.
    """
    samples = torch.cat([sample_gaussian(m, s, n) for m, s, n in zip(means, covars, n_samples)])
    samples = samples[torch.randperm(len(samples))]
    return samples

def load_image(fname):
    """Loads an image, transform it into gray values, and reverse it to have black pixels
        represented by 1 and white pixels represented by 0.

    Args:
        fname (str | Path | BinaryIO): path of the image.

    Returns:
        ndarray: image as a numpy array, with shape (h, w) and values between 0 and 1.
    """
    img = np.mean(imread(fname), axis=2)  # Grayscale
    img = (img[:, :]) / 255.0
    return 1 - img  # black = 1, white = 0

def cloud_transform(image, n_samples=1000):
    """Transform a gray-scaled image into a cloud of points.

    Args:
        image (ndarray): shape (h, w). Non-zero locations are those that will be sampled.
        n_samples (int, optional): Number of samples. Defaults to 1000.

    Returns:
        ndarray: shape (n_samples, 2). Samples are taken among the nonzero locations of image,
            after which some noise is added.
    """
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
    def __init__(self, dataset_name, batch_size_train, batch_size_test, **kwargs):
        self.dataset_name = dataset_name
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.kwargs = kwargs

    def load_data(self):

        if self.dataset_name == "gaussians":
            n_train = self.kwargs["n_train"] if "n_train" in self.kwargs else 1000
            n_test  = self.kwargs["n_test" ] if "n_test"  in self.kwargs else 1000
            n_val   = self.kwargs["n_val"  ] if "n_val"   in self.kwargs else 1000

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
            n_train = self.kwargs["n_train"] if "n_train" in self.kwargs else 1000
            n_test  = self.kwargs["n_test" ] if "n_test"  in self.kwargs else 1000
            n_val   = self.kwargs["n_val"  ] if "n_val"   in self.kwargs else 1000

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

        elif self.dataset_name == "obelix" or self.dataset_name == "asterix":
            n_train = self.kwargs["n_train"] if "n_train" in self.kwargs else 5000
            n_test  = self.kwargs["n_test" ] if "n_test"  in self.kwargs else 5000
            n_val   = self.kwargs["n_val"  ] if "n_val"   in self.kwargs else 5000

            train_x = torch.tensor(cloud_transform(load_image(DATA_FOLDER / f"{self.dataset_name}.jpg"), n_samples=n_train)).float()
            val_x   = torch.tensor(cloud_transform(load_image(DATA_FOLDER / f"{self.dataset_name}.jpg"), n_samples=n_val  )).float()
            test_x  = torch.tensor(cloud_transform(load_image(DATA_FOLDER / f"{self.dataset_name}.jpg"), n_samples=n_test )).float()

            train_dataset = TensorDataset(train_x, torch.zeros(len(train_x)))
            val_dataset   = TensorDataset(val_x  , torch.zeros(len(val_x  )))
            test_dataset  = TensorDataset(test_x , torch.zeros(len(test_x )))

            train_loader = DataLoader(train_dataset, batch_size = self.batch_size_train, shuffle=True)
            val_loader   = DataLoader(val_dataset  , batch_size = self.batch_size_test , shuffle=True)
            test_loader  = DataLoader(test_dataset , batch_size = self.batch_size_test , shuffle=True)
            
        data_loaders = {'train': train_loader,
                        'test': test_loader, 'val': val_loader}

        return data_loaders
    


