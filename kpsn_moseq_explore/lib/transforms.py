from jax_moseq.models.keypoint_slds import alignment
import jax.numpy as jnp
from ..lib import skeleton


def align_root(
    keypts,
    config
    ):
    """
    keypts: array (frames, keypts, spatial)"""
    arm = skeleton.Armature.from_config(config)
    root = keypts[..., arm.keypt_by_name[arm.root], None, :]
    return keypts - root, root

def invert_align_root(
    keypts,
    root,
    config
    ):
    return keypts + root


def scale_keypoint_array(
    keypts,
    factor,
    anterior_idxs,
    posterior_idxs,
    **kwargs
    ):
    
    keypts = jnp.array(keypts)
    Y_aligned, v, h = alignment.align_egocentric(keypts, anterior_idxs, posterior_idxs)
    
    Y_scaled = Y_aligned * factor

    return alignment.rigid_transform(Y_scaled, v, h)


def scalar_align(
    keypts,
    config
    ):
    """
    Parameters
    ----------
    keypts : dict[str, numpy array (frames, keypts, spatial)]"""

    align_data = {
        # [y_centered, roots]
        s: align_root(jnp.array(kpts), config)
        for s, kpts in keypts.items()}
    
    absolute_scales = {}
    for s, kpts in keypts.items():
        anterior_com = kpts[:, config['anterior_idxs']].mean(axis = 1)
        posterior_com = kpts[:, config['posterior_idxs']].mean(axis = 1)
        absolute_scales[s] = jnp.median(
            jnp.linalg.norm(anterior_com - posterior_com, axis = -1),
        axis = 0)
    
    mean_scl = sum(absolute_scales.values()) / len(absolute_scales)
    # print(mean_scl)
    scales = {s: scl / mean_scl for s, scl in absolute_scales.items()}
    # scales = {s: 1 for s, scl in absolute_scales.items()}
    # # print(scales)
    scaled_keypts = {
        s: align_data[s][0] / scales[s]
        for s in align_data}

    unaligned = {
        s: invert_align_root(scaled_keypts[s], align_data[s][1], config)
        # s: scaled_keypts[s]
        # s: keypts[s]
        for s in align_data}
    return unaligned