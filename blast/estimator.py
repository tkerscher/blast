import numpy as np
import onnxruntime as rt

from importlib.resources import as_file, files
from scipy.spatial import KDTree

_bin_mean = None
_bin_scale = None
_label_mean = None
_label_scale = None
with as_file(files(__package__).joinpath("scaling.npz")) as p:
    scaling = np.load(p)
    _bin_mean = scaling['bin_mean']
    _bin_scale = scaling['bin_scale']
    _label_mean = scaling['label_mean']
    _label_scale = scaling['label_scale']

_bins = np.loadtxt(files("blast").joinpath("bins.txt"), skiprows=1)
_freq_inc = _bins[:,2].astype(np.bool_)
_flux_inc = _bins[:,3].astype(np.bool_)

_positions = None
_bags = None
with as_file(files(__package__).joinpath("bag_index.npy")) as p:
    index = np.load(p)
    _positions = index[:,:2]
    _bags = index[:,2].astype(np.int32)
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


class Ensemble():
    """
    Class managing an ensemble of neural networks taking a binned sed and
    producing an estimate with an prediction interval.
    """  

    def __init__(self, model_dir, bin_mask, label_mean, label_scale) -> None:
        # load models
        self._models = [
            [rt.InferenceSession(model_dir / f"{bag}.{i}.onnx") for i in range(5)]
            for bag in range(5)
        ]
        # save params
        self._bin_mask = bin_mask
        self._label_mean = label_mean
        self._label_scale = label_scale
    
    def __call__(self, sed, bag=None, *, sigma=1.96):
        #sed is expected to be a allready binned, thus a numpy array of shape (N,30)
        if not type(sed) is np.ndarray:
            raise ValueError('Expected sed to be a numpy array!')
        #Add batch dimension if not present
        if sed.ndim == 1:
            sed = sed.reshape((1,-1))
        #check dimension
        if sed.ndim > 2 or sed.shape[1] != len(_bins):
            raise ValueError('Expected sed to be numpy array of shape (N,30)!')
        
        # prepare data
        mask = (sed != 0.0).astype(np.float32)
        sed = (sed - _bin_mean) / _bin_scale * mask
        sed, mask = sed[:,self._bin_mask], mask[:,self._bin_mask]
        x = np.concatenate([sed, mask], axis=1).astype(np.float32)
        args = (["mean", "var"], {"bins": x})

        models = None
        #check for out of bag estimate
        if bag is not None and bag >= 0 and bag < len(self._models):
            models = self._models[bag]
        else:
            models = [bag[i] for bag in self._models for i in range(5)]
        # run models
        results = np.stack([np.array(m.run(*args)).T for m in models], axis=0)
        mean = np.mean(results[...,0], 0)
        var = np.mean(results[...,1] + np.square(mean), 0) - np.square(mean)
        std = np.sqrt(var)

        # scale labels back
        mean = mean * self._label_scale + self._label_mean
        err = sigma * std * self._label_scale

        return mean.squeeze(), err.squeeze()


class PeakFrequencyEstimator(Ensemble):
    """
    Estimator for the frequency of the synchrotron peak of blazars.
    To get an estimation the estimator can simply be called with a binned sed::

        from blast import PeakFrequencyEstimator, bin_data, parse_sed

        estimator = PeakFrequencyEstimator()
        sed = bin_data(parse_sed('sed.txt'))
        peak, err = estimator(sed)

    The estimator returns both an estimate for the synchrotron peak as well as a
    prediction interval of given width (default is 95%). It is also possible to
    batch multiple seds along the first axis. To specify a different prediction
    interval width, supply the named sigma argument::

        peak, err = estimator(sed, sigma=1.0)
    
    """

    def __init__(self):
        super().__init__(
            files(__package__).joinpath("models/freq/"),
            _freq_inc,
            _label_mean[0],
            _label_scale[0]
        )


class PeakFluxEstimator(Ensemble):
    """
    Estimator for the flux of the synchrotron peak of blazars.
    To get an estimation the estimator can simply be called with a binned sed::

        from blast import PeakFluxEstimator, bin_data, parse_sed

        estimator = PeakFluxEstimator()
        sed = bin_data(parse_sed('sed.txt'))
        peak, err = estimator(sed)

    The estimator returns both an estimate for the synchrotron peak as well as a
    prediction interval of given width (default is 95%). It is also possible to
    batch multiple seds along the first axis. To specify a different prediction
    interval width, supply the named sigma argument::

        peak, err = estimator(sed, sigma=1.0)
    
    """

    def __init__(self):
        super().__init__(
            files(__package__).joinpath("models/flux/"),
            _flux_inc,
            _label_mean[1],
            _label_scale[1],
        )
