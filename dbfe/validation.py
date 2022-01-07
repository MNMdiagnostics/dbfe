import numpy as np

from sklearn.model_selection import KFold

def check_n_bins(n_bins):
    result = None

    if n_bins in ['auto', 'all', 'sd']:
        result = n_bins
    elif isinstance(n_bins, int):
        result = n_bins
    elif isinstance(n_bins, float):
        result = n_bins
    else:
        raise ValueError('Unsupported value. Only \"auto\", \"all\", \"sd\", float, and int are supported.')

    return result
    
def check_y(y):
    if not all(np.unique(y) == [0, 1]):
        raise ValueError('Labels must consist of exactly 2 classes: 0 and 1!')

def check_values(values):
    if not all(values > 0):
        raise ValueError('All values must be greater than 0!')

def check_cv(cv):
    result = None

    if isinstance(cv, int):
        result = KFold(cv)
    elif callable(getattr(cv, "split", None)):
        result = cv
    elif cv is None:
        result = cv
    else:
        raise ValueError('Unsupported value. Only int, BaseCrossValidator, and None are supported.')

    return result
