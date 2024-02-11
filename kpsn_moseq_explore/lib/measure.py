import numpy as np
from scipy.spatial import distance

def syll_counts(
    arr,
    sylls,
    ):
    """
    Count occurences in an array
    """
    uniq, counts = np.unique(arr, return_counts = True)
    
    return np.array([
        counts[uniq == s][0]
        if np.isin(s, uniq) else 0
        for s in sylls
    ])

def jsd(
    counts1, counts2
    ):
    """
    Compute Jensen-Shannon distance between count arrays.

    Performed with Dirichlet prior to handle zero-counts.
    """
    
    pmf1 = (counts1 + 1) / (np.sum(counts1) + counts1.size)
    pmf2 = (counts2 + 1) / (np.sum(counts2) + counts2.size)
    return distance.jensenshannon(pmf1, pmf2)