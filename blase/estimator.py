import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib.resources as res

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

    The estimator returns both an estimate for the synchrotron peak as well as a 95% prediction
    interval. It is also possible to batch multiple seds along the first axis.
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

    def __call__(self, sed, bag=None):
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
        for i in range(5):
            #skip if sed is part of bag
            if i == bag:
                continue
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
        err = 1.96 * std * self.label_scale

        return mean.squeeze(), err.squeeze()
