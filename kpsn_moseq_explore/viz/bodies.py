import keypoint_moseq.util
import matplotlib.pyplot as plt
import jax.numpy as jnp

def plot_mouse_2d(
    ax,
    keypt_frame,
    xaxis,
    yaxis,
    bodyparts,
    use_bodyparts,
    skeleton,
    color = None,
    scatter_kw = {},
    line_kw = {},
    **kwargs):

    edges = keypoint_moseq.util.get_edges(use_bodyparts, skeleton)
    use_keypts = jnp.stack([keypt_frame[bodyparts.index(bp)] for bp in use_bodyparts])

    ax.scatter(use_keypts[..., xaxis], use_keypts[..., yaxis],
        **{'color': color, 's': 3, **scatter_kw})
    
    for child, parent in edges:
        curr_child = use_keypts[child]
        curr_parent = use_keypts[parent]
        ax.plot((curr_child[xaxis], curr_parent[xaxis]),
                (curr_child[yaxis], curr_parent[yaxis]),
                **{'color':color, 'lw': 1, **line_kw})
        
    ax.set_aspect(1.)