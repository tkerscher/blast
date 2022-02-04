import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib.resources as res
from scipy.spatial import KDTree

_positions = None
_bags = None
with res.path(__package__, 'bag_index.npy') as p:
    index = np.load(p)
    _positions = index[:,:2]
    _bags = index[:,2].astype(int)
_tree = KDTree(_positions)
#KDTree returns one after the last as index for not found
#We exploit that by adding a 'not found'-value at the end of bags
_bags = np.append(_bags, -1)

def get_bag(positions, radius=0.1) -> np.ndarray:
    """
    Returns the bag index for the seds at the given positions. Seds that were
    not part of the training set are given the index -1.

    Parameters:
    positions (numpy.ndarray): The positions of the seds to query. Either of
    shape (2,) for a single sed or (N,2) for a batch.

    Returns:
    The bag index of the seds. Either a single int if only one sed was queried
    or a numpy.ndarray of type int and shape (N,). -1 denotes unseen data.
    """
    _, indices = _tree.query(positions, distance_upper_bound=radius)
    return _bags[indices]

class Model(nn.Module):
    """
    Individual ensemble member of the estimator. Direct use of this class is
    most likely not intended as it's expecting input as pytorch tensor.
    """
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(52, 152),
            nn.BatchNorm1d(152),
            nn.Dropout(0.11623816061109485),
            nn.ReLU(),
            nn.Linear(152, 80),
            nn.BatchNorm1d(80),
            nn.Dropout(0.14953177977171542),
            nn.ReLU(),
            nn.Linear(80, 72),
            nn.BatchNorm1d(72),
            nn.Dropout(0.024569432237666035),
            nn.ReLU(),
            nn.Linear(72, 48),
            nn.BatchNorm1d(48),
            nn.Dropout(0.03208157605345701),
            nn.ReLU(),
            nn.Linear(48, 2)
        )
    def forward(self, X):
        out = self.model(X)
        #the second output is the variance
        #use softplus to force the variance in [0,inf]
        mean, var = torch.unbind(out, dim=1)#first axis is batch
        var = F.softplus(var) #enforce > 0
        return mean, var

class Estimator():
    r"""
    Estimator for synchrotron peaks of blazars. To get an estimation the estimator can simply be
    called with a binned sed::

        from blase import Estimator, bin_data, parse_sed

        estimator = Estimator()
        sed = bin_data(parse_sed('sed.txt'))
        peak, err = estimator(sed)

    The estimator returns both an estimate for the synchrotron peak as well as a prediction
    interval of given width (default is 95%). It is also possible to batch multiple seds along the
    first axis. To specify a different prediction interval width, supply the named sigma argument::

        peak, err = estimator(sed, sigma=1.0)
    
    """
    def __init__(self):
        #load scaling
        with res.path(__package__, 'scaling.npz') as p:
            scaling = np.load(p)
            self.bin_mean = scaling['bin_mean']
            self.bin_scale = scaling['bin_scale']
            self.label_mean = scaling['label_mean']
            self.label_scale = scaling['label_scale']
        #load models
        def load_model(models, bag, i):
            model = Model()
            model.load_state_dict(models[f'{bag}.{i}'])
            model.eval()
            return model
        with res.path(__package__, 'models.pth') as p:
            models = torch.load(p)
            self.models = [[load_model(models, bag, i) for i in range(5)] for bag in range(5)]      

    def __call__(self, sed, bag=None, *, sigma=1.96):
        #sed is expected to be a allready binned, thus a numpy array of shape (N,26)
        if not type(sed) is np.ndarray:
            raise ValueError('Expected sed to be a numpy array!')
        #Add batch dimension if not present
        if sed.ndim == 1:
            sed = sed.reshape((1,-1))
        #check dimension
        if sed.ndim > 2 or sed.shape[1] != 26:
            raise ValueError('Expected sed to be numpy array of shape (N,26)!')
        
        #prepare data
        mask = (sed != 0.0).astype(float)
        sed = (sed - self.bin_mean) / self.bin_scale * mask
        input = np.concatenate((sed, mask), axis=1)
        input = torch.tensor(input).float()

        means = []
        vars = []
        #check for out of bag estimate
        if bag is not None and bag >= 0 and bag < len(self.models):
            for member in self.models[bag]:
                mean, var = member(input)
                means.append(mean)
                vars.append(var)
        else:
            for i in range(5):
                for member in self.models[i]:
                    mean, var = member(input)
                    means.append(mean)
                    vars.append(var)
        means = torch.stack(means)
        vars = torch.stack(vars)

        mean = torch.mean(means,0)
        var = torch.mean(vars + torch.square(means), 0) - torch.square(mean)
        std = torch.sqrt(var)

        #fetch data from gpu
        mean = mean.detach().cpu().numpy()
        std = std.detach().cpu().numpy()

        #scale back
        mean = mean * self.label_scale + self.label_mean
        err = sigma * std * self.label_scale

        return mean.squeeze(), err.squeeze()
