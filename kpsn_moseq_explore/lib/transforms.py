from jax_moseq.models.keypoint_slds import alignment
import jax.numpy as jnp

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


    