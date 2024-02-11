import numpy as np
import glob
import re
import joblib as jl
import keypoint_moseq as kpms
import jax_moseq.utils
from .kpms_custom_io import (
    load_keypoints, modata_name_func, create_multicam_gimbal_loader, load_glob, 
    get_groups_dict, modata_id_from_sess_name, modata_age_from_sess_name,
    with_age)
from .transforms import scale_keypoint_array, scalar_align
from ..lib import linear_skeletal as ls
from ..lib import skeleton



def _wrap_apply_format(func):
    def new_func(name, dir, config, format = True):
        coords, confs = func(name, dir, config)
        if format:
            data, metadata = kpms.format_data(coords, confs, **config)
            data = jax_moseq.utils.convert_data_precision(data)
            return data, metadata
        else:
            return coords, confs
    return new_func


# Select few animals across a spectrum of scales (by gross rigid transform)
# -----------------------------------------------------------------------------

@_wrap_apply_format
def multiscale(dataset_name, data_dir, config):
    match = re.search(r"multiscale_wk(\d+)_m(\d+)", dataset_name)
    age = match.group(1)
    subj_ids = list(match.group(2))
    
    filenames = glob.glob(f'{data_dir}/**/*.gimbal_results.p', recursive = True)
    filenames = [f for f in filenames if (
        modata_age_from_sess_name(modata_name_func(f)) == age and
        modata_id_from_sess_name (modata_name_func(f)) in subj_ids)]
    
    coordinates, confidences, _ = load_keypoints(
        filenames,
        create_multicam_gimbal_loader(config['bodyparts']),
        name_func = modata_name_func)
    
    scale_factors = [0.7, 0.85, 1.15, 1.3]
    scaled_coordinates = {}
    scaled_confidences = {}
    for sess, coords in coordinates.items():
        scaled_coordinates[f'{sess}_f1'] = coords
        scaled_confidences[f'{sess}_f1'] = confidences[sess]
        for scale_factor in scale_factors:
            scaled_coordinates[f'{sess}_f{scale_factor}'] = np.array(
                scale_keypoint_array(coords, scale_factor, **config))
            scaled_confidences[f'{sess}_f{scale_factor}'] = confidences[sess]
    print(f"Generated scaled sessions.")
            
    return coordinates, confidences


# Mice from a cohort at at two sizes by gross rigid transform
# -----------------------------------------------------------------------------

@_wrap_apply_format
def twoscale(dataset_name, data_dir, config):

    match = re.search(r"twoscale_wk(\d+)", dataset_name)
    age = match.group(1)
    print(f"Cohort: {age}wk")
    
    filenames = glob.glob(f'{data_dir}/**/*.gimbal_results.p', recursive = True)
    filenames = [f for f in filenames if 
        modata_age_from_sess_name(modata_name_func(f)) == age]
    
    coordinates, confidences, _ = load_keypoints(
        filenames,
        create_multicam_gimbal_loader(config['bodyparts']),
        name_func = modata_name_func)
    
    print(f"Sessions (pre scale): {list(coordinates.keys())}")

    scale_factor = 1.3
    coordinates = {
        **{f'{sess}_f{scale_factor}':
            np.array(scale_keypoint_array(
                coords, scale_factor, **config))
            for sess, coords in coordinates.items()},
        **{f'{sess}_f1': coords
            for sess, coords in coordinates.items()}
    }
    confidences = {
        **{f'{sess}_f{scale_factor}': conf
            for sess, conf in confidences.items()},
        **{f'{sess}_f1': conf
            for sess, conf in confidences.items()}
    }
    print(f"Generated scaled sessions f{scale_factor}")

    return coordinates, confidences


# Bone lengths scaled as appropriate to target age
# -----------------------------------------------------------------------------

@_wrap_apply_format
def blscale(dataset_name, data_dir, config):
    match = re.search(r"blscale_wk(\d+)_to(\d+)", dataset_name)
    src_age = match.group(1)
    tgt_age = match.group(2)
    print(f"Cohort: {src_age}wk")
    print(f"Target: {tgt_age}wk")

    src_sessions, (src_coords, src_confs) = with_age(data_dir, src_age, config)
    tgt_sessions, (tgt_coords, tgt_confs) = with_age(data_dir, tgt_age, config)
    coordinates = {**src_coords, **tgt_coords}
    confidences = {**src_confs, **tgt_confs}

    roots, bones, (arm, ls_mat) = ls.roots_and_bones(coordinates, config)
    subj_lengths = {
        s: np.linalg.norm(bones[s], axis = -1).mean(axis = 0)
        for s in coordinates}
    tgt_lengths = np.mean([
        subj_lengths[s] for s in tgt_sessions], axis = 0)

    # create new copies of the these scaled by age
    new_bones = {
        s: np.array(bones[s] * (tgt_lengths / subj_lengths[s])[None, :, None])
        for s in src_sessions}
    new_kpts = {
        s: ls.inverse_transform(roots[s], new_bones[s], ls_mat)
        for s in src_sessions}
    
    # assemble dataset of original and rescaled keypoints
    ids = {s: modata_id_from_sess_name(s) for s in src_sessions}
    new_coords = {
        **{f'm{ids[sess]}:{tgt_age}wk': new_kpts[sess]
            for sess in src_sessions},
        **{f'm{ids[sess]}:{src_age}wk': coordinates[sess]
            for sess in src_sessions}
    }
    new_confs = {
        **{f'm{ids[sess]}:{tgt_age}wk': confidences[sess]
            for sess in src_sessions},
        **{f'm{ids[sess]}:{src_age}wk': confidences[sess]
            for sess in src_sessions}
    }

    new_coords = scalar_align(new_coords, config)
    new_coords = {s: np.array(c) for s, c in new_coords.items()}
    
    return new_coords, new_confs


# Mo's unaltered dataset
# -----------------------------------------------------------------------------

def modata(data_dir, config, format = True):
    coordinates, confidences, bodyparts = load_glob(
        f'{data_dir}/**/*.gimbal_results.p',
        create_multicam_gimbal_loader(config['bodyparts']),
        name_func = modata_name_func)
    
    # format data for modeling
    if format:
        data, metadata = kpms.format_data(coordinates, confidences, **config)
        data = jax_moseq.utils.convert_data_precision(data)
    
        return data, metadata
    else:
        return coordinates, confidences
    
