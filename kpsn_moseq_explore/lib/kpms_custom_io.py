from keypoint_moseq import util as kpms_util
from keypoint_moseq import io as kpms_io
from textwrap import fill
import joblib as jl
import numpy as np
import tqdm
import os.path

def _name_from_path(filepath, path_in_name, path_sep, remove_extension):
    """Create a name from a filepath.

    Either return the name of the file (with the extension removed) or return
    the full filepath, where the path separators are replaced with `path_sep`.
    """
    if remove_extension:
        filepath = os.path.splitext(filepath)[0]
    if path_in_name:
        return filepath.replace(os.path.sep, path_sep)
    else:
        return os.path.basename(filepath)


def load_keypoints_with_loader(
    filepath_pattern,
    loader,
    recursive=True,
    path_sep="-",
    path_in_name=False,
    remove_extension=True,
    name_func=None
):
    """
    Returns
    -------
    coordinates: dict
        Dictionary mapping filenames to keypoint coordinates as ndarrays of
        shape (n_frames, n_bodyparts, 2[or 3])

    confidences: dict
        Dictionary mapping filenames to `likelihood` scores as ndarrays of
        shape (n_frames, n_bodyparts)
    bodyparts: list of str
        List of bodypart names. The order of the names matches the order of the
        bodyparts in `coordinates` and `confidences`.
    """

    import glob
    filepaths = glob.glob(filepath_pattern, recursive = True)
    assert len(filepaths) > 0, fill(
        f"No such files {filepath_pattern}"
    )

    coordinates, confidences, bodyparts = {}, {}, None
    for filepath in tqdm.tqdm(filepaths, desc=f"Loading keypoints", ncols=72):
        try:
            name = (_name_from_path if name_func is None else name_func)(
                filepath, path_in_name, path_sep, remove_extension
            )
            new_coordinates, new_confidences, bodyparts = loader(
                filepath, name
            )

            if set(new_coordinates.keys()) & set(coordinates.keys()):
                raise ValueError(
                    f"Duplicate names found in {filepath_pattern}:\n\n"
                    f"{set(new_coordinates.keys()) & set(coordinates.keys())}"
                    f"\n\nThis may be caused by repeated filenames with "
                    "different extensions. If so, please set the extension "
                    "explicitly via the `extension` argument. Another possible"
                    " cause is commonly-named files in different directories. "
                    "if that is the case, then set `path_in_name=True`."
                )

        except Exception as e:
            print(fill(f"Error loading {filepath}: {e}"))

        coordinates.update(new_coordinates)
        confidences.update(new_confidences)

    assert len(coordinates) > 0, fill(
        f"No valid results found for {filepath_pattern}"
    )

    kpms_util.check_nan_proportions(coordinates, bodyparts)
    return coordinates, confidences, bodyparts


def create_multicam_gimbal_loader(bodyparts):
    def multicam_gimbal_loader(filepath, name):
        # (n_frames, n_bodyparts, n_dim)
        coords = jl.load(filepath)['positions_medfilter']
        confs = np.ones_like(coords[..., 0])
        return {name: coords}, {name: confs}, bodyparts
    return multicam_gimbal_loader