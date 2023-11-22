import numpy as np
import glob
import re
import keypoint_moseq as kpms
import jax_moseq.utils
from .kpms_custom_io import load_keypoints, modata_name_func, create_multicam_gimbal_loader, load_glob
from .transforms import scale_keypoint_array

modata_age_from_sess_name = lambda name: name.split('w')[0]
modata_id_from_sess_name  = lambda name: name.split('m')[1]

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


@_wrap_apply_format
def twoscale(dataset_name, data_dir, config):

    coordinates, confidences, bodyparts = load_glob(
            f'{data_dir}/**/*.gimbal_results.p',
            create_multicam_gimbal_loader(config['bodyparts']),
            name_func = modata_name_func)
    
    cohort = dataset_name.split('k')[1]
    print(f"Cohort: {cohort}wk")
    coordinates = {sess: coords for sess, coords in coordinates.items()
                   if sess.split('w')[0] == cohort}
    confidences = {sess: confidences[sess] for sess in coordinates}
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


def modata(data_dir, config):
    coordinates, confidences, bodyparts = load_glob(
        f'{data_dir}/**/*.gimbal_results.p',
        create_multicam_gimbal_loader(config['bodyparts']),
        name_func = modata_name_func)
    
    # format data for modeling
    data, metadata = kpms.format_data(coordinates, confidences, **config)
    data = jax_moseq.utils.convert_data_precision(data)
    
    return data, metadata
    
