import numpy as np
from scipy.spatial import distance
from scipy.cluster import hierarchy
import jax.tree_util as pt

def tree_unique(tree):
    return pt.tree_reduce(lambda a, b:
        np.unique(np.concatenate([np.unique(a), np.unique(b)])),
        tree)

def usage_matrix(
    original_labels,
    perturbed_labels,
    label_set,
    order_metric = 'correlation'):

    confusion = confusion_matrix(original_labels, perturbed_labels, label_set)

    # no nans in distances
    dist_eps = confusion.std() * 1e-3
    dist_noise = np.random.RandomState(0).normal(
        size = confusion.shape) * dist_eps
    # agglomerative cluster to permute rows/cols
    dists = distance.pdist(
        confusion + dist_noise,
        metric = order_metric)
    ward_matrix = hierarchy.ward(dists)
    ordering = hierarchy.leaves_list(
        hierarchy.optimal_leaf_ordering(ward_matrix, dists))
    ordered_confusion = confusion[ordering, :][:, ordering]

    return ordered_confusion, ordering


def confusion_matrix(
    original_labels,
    perturbed_labels,
    label_set):

    confusion = np.array([[
            # intersection usage / original usage
            (((original_labels == orig_syll) &
              (perturbed_labels == new_syll)).sum())
        for new_syll in label_set]
        for orig_syll in label_set]).astype('float')

    row_sums = confusion.sum(axis = 1)
    confusion /= np.where(row_sums != 0., row_sums, 1.)[:, None]
    
    return confusion
