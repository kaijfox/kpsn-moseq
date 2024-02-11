from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import numpy as np

def count_dict_to_feats_labels(counts, groups):
    """
    counts : dict[str, array (n_feat,)]
    groups : iterable[iterable[str]]
    """
    # assign integer to each group
    labels = []
    X = []
    for i_grp, grp_members in enumerate(groups):
        for member in grp_members:
            labels.append(i_grp)
            X.append(counts[member])
    return np.stack(X), np.stack(labels)


def whiten(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, scaler


def svc_crossval_scores(X, labels, cls_kw = {}):
    mask = np.ones(len(X), dtype = bool)
    scores = np.zeros(len(X))
    for i in range(len(X)):
        mask[...] = 1
        mask[i] = 0
        svm = LinearSVC(dual = True, class_weight = 'balanced', **cls_kw
                        ).fit(X[mask], labels[mask])
        scores[i] = svm.decision_function(X[[i]])[0]
    return scores


def svc_paired_diffs(X0, X1, cls_kw = {}):
    """
    X0, X1: array (n_pairs, n_feats)
    """
    mask = np.ones(len(X0), dtype = bool)
    labels = np.concatenate([
        np.zeros(len(X0) - 1, dtype = int),
        np.ones(len(X1) - 1, dtype = int)])
    scores = np.zeros([2, len(X0)])
    for i in range(len(X0)):
        mask[...] = 1
        mask[i] = 0
        X = np.concatenate([X0[mask], X1[mask]])
        svm = LinearSVC(dual = True, max_iter = 10000, **cls_kw).fit(X, labels)
        scores[:, i] = svm.decision_function(np.stack([X0[i], X1[i]]))
    return scores