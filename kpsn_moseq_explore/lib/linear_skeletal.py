import numpy as np
import keypoint_moseq as kpms
from ..lib import skeleton


def construct_transform(skeleton, root_keypt):
    n_kpts = len(skeleton) + 1
    u_to_x = np.zeros([n_kpts, n_kpts])
    x_to_u = np.zeros([n_kpts, n_kpts])
    u_to_x[root_keypt, root_keypt] = 1
    x_to_u[root_keypt, root_keypt] = 1
    # skeleton is topo sorted (thank youuuu)
    for (child, parent) in skeleton:
        x_to_u[child, parent] = -1
        x_to_u[child, child] = 1
        u_to_x[child] = u_to_x[parent]
        u_to_x[child, child] = 1
    bones_mask = np.ones(n_kpts, dtype = bool)
    bones_mask[root_keypt] = 0
    return {"u_to_x": u_to_x, "x_to_u": x_to_u,
            'root': root_keypt, 'bone_mask': bones_mask}


def transform(keypts, transform_data):
    bones_and_root = transform_data['x_to_u'] @ keypts
    return (bones_and_root[..., transform_data['root'], :],
            bones_and_root[..., transform_data['bone_mask'], :])


def join_with_root(bones, roots, transform_data):
    return np.insert(
        bones, transform_data['root'], roots,
        axis = -2)  


def inverse_transform(roots, bones, transform_data):
    if roots is None:
        roots = np.zeros(bones.shape[:-2] + (bones.shape[-1],))
    bones_and_root = join_with_root(bones, roots, transform_data)
    return transform_data['u_to_x'] @ bones_and_root


def roots_and_bones(coords, config):
    # create skeleton
    arm = skeleton.Armature.from_config(config)
    ls_mat = construct_transform(arm.bones, arm.keypt_by_name[arm.root])

    # measure bone lengths by age
    roots_and_bones = {s: transform(coords, ls_mat)
        for s, coords in coords.items()}
    roots = {s: roots_and_bones[s][0] for s in coords}
    bones = {s: roots_and_bones[s][1] for s in coords}
    
    return roots, bones, (arm, ls_mat)