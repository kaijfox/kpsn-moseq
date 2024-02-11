import numpy as np

def stripgroups(names, arrs):
    x = np.concatenate([np.full(arr.shape, name) for name, arr in zip(names, arrs)])
    y = np.concatenate([arr for arr in arrs])
    return dict(x = x, y = y)