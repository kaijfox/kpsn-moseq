from typing import NamedTuple
import numpy as np
import jax.numpy as jnp
import keypoint_moseq.util as kpms_util


# Define armature
# -----------------------------------------------------------------------------

class Armature(NamedTuple):
    keypt_names: np.ndarray #<str>
    bones: np.ndarray #<int>
    root: str

    @property
    def keypt_by_name(self):
        return {n: i for i, n in enumerate(self.keypt_names)}
    
    @property
    def n_kpts(self): return len(self.keypt_names)

    def bone_name(self, i_bone, joiner = '-'):
        child_name = self.keypt_names[self.bones[i_bone, 0]]
        parent_name = self.keypt_names[self.bones[i_bone, 1]]
        return f'{child_name}{joiner}{parent_name}'
    
    @staticmethod
    def from_config(config):
        skel_ixs = kpms_util.get_edges(config['bodyparts'], config['skeleton'])
        root_ix = [
            i_kpt for i_kpt in range(len(config['bodyparts']))
            if i_kpt not in np.array(skel_ixs)[:, 0]][0]
        return Armature(
            keypt_names = np.array(config['bodyparts']),
            bones = np.array(skel_ixs),
            root = config['bodyparts'][root_ix])
        





# Skeleton traversal and manipulation
# -----------------------------------------------------------------------------


def reroot(skel, new_root):
    new_bones = []

    def traverse_from(node):
        visited.add(node)
        for child in connected_to(node):
            if child in visited: continue
            new_bones.append((child, node))
            traverse_from(child)
    
    visited = set()
    def connected_to(i):
        return np.concatenate([
            skel.bones[skel.bones[:, 0] == i, 1],
            skel.bones[skel.bones[:, 1] == i, 0]])

    traverse_from(skel.keypt_by_name[new_root])
    return Armature(
        keypt_names = skel.keypt_names,
        bones = np.array(new_bones),
        root = new_root)




