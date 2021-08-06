import importlib.resources as res
import numpy as np
from os.path import isfile

_bin_edges = None
with res.path(__package__, 'bins.txt') as p:
    _bin_edges = np.loadtxt(p)

def get_bin_edges() -> np.ndarray:
    """
    Returns the bin edges used for binning noted in log10.
    Returned numpy array has shape (N,2) where N is the number of bins.
    """
    return _bin_edges

def bin_data(sed) -> np.ndarray:
    """
    Bins the given sed.

    Parameters:
    sed (numpy.ndarray): The sed of shape (N,2) like it's outputted by parse_sed.

    Returns:
    The binned sed of shape (N,26).
    """
    def bin(sed):
        line = []
        for bin in _bin_edges:
            inside = (sed[:,0] >= bin[0]) & (sed[:,0] <= bin[1])
            flux = sed[inside][:,1]
            line.append(np.mean(flux) if len(flux) > 0 else 0.0)
        return line
    
    if type(sed) is np.ndarray and sed.ndim == 2 and sed.shape[1] == 2:
        return np.array(bin(sed))
    else:
        try:
            return np.ndarray([bin(s) for s in sed])
        except TypeError:
            raise ValueError('Expected either a single sed as numpy array of shape (N,2) or a list of such seds')

def parse_sed(sed, sanitize=True) -> np.ndarray:
    """
    Parses a sed given by either a filepath or the string containing it.
    The result has a shape of (N,2), i.e. each row is a sample while the first column is the
    frequency in Hz and the second the flux in erg/cm²/s both in log10 space.

    Parameters:
    sed: Either filepath or string of a sed.
    sanitize: True if nonsense data such as zero flux should be ignore. Default True.

    Returns:
    Sed in log10 of shape (N,2) with each row a sample point and the columns the frequency and flux.
    """
    def parse(lines, sanitize):
        _data = []
        for line in lines[4:]:
            entries = line.split()
            x = float(entries[0])
            y = float(entries[1])
            up = float(entries[2])
            lo = float(entries[3])
            #sanity check errors
            if sanitize and (up < y or y < lo) and up != 0.0 and lo != 0.0:
                continue #Skip this entry
            _data.append([x, y])
        _data = np.array(_data)
        if sanitize:
            _data = np.delete(_data, _data[:,0] <= 0, axis=0)
            _data = np.delete(_data, _data[:,1] <= 0, axis=0)
        _data = np.log10(_data)
        return _data

    if isfile(sed):
        with open(sed, 'r') as f:
            return parse(f.readlines(), sanitize)
    elif type(sed) is str:
        return parse(sed.split('\n'), sanitize)
    else:
        raise ValueError('Expected sed to be either a path to a sed file or a string containing it')
