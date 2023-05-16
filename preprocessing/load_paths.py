import os
import numpy as np


def load_prepared_img_paths(data_dir_path: str):
    paths = []
    folders = os.listdir(data_dir_path)
    for index in range(len(folders)):
        for item in os.listdir(os.path.join(data_dir_path, folders[index])):
            paths.append(os.path.join(data_dir_path, folders[index], item))
    return paths


def load_raw_data_paths(data_dir_path: str):
    folders = os.listdir(data_dir_path)
    mri_type_list = ["t1", "t1ce", "t2", "flair"]
    path_list = []
    mask_list = []
    # iterate over folders in data_dir
    for index in range(len(folders)):
        channels = []
        channels = {
            "t1": "",
            "t1ce": "",
            "t2": "",
            "flair": "",
        }

        # load path of every channel and set to map
        for mri_type in mri_type_list:
            channels[mri_type] = __get_file_path(
                data_dir_path, folders, index=index, mri_type=mri_type
            )

        path_list.append(channels)
        mask_list.append(
            __get_file_path(data_dir_path, folders, index=index, mask=True)
        )

    return np.array(path_list), np.array(mask_list)


def __get_image_name(folders, index: int, mri_type: str):
    return folders[index] + "_" + mri_type + ".nii.gz"


def __get_label_image_name(folders, index: int):
    return folders[index] + "_seg" + ".nii.gz"


def __get_file_path(
    data_dir_path: str, folders, index: int, mri_type="t1", mask: bool = False
):
    if mask:
        name = __get_label_image_name(folders, index)
    else:
        name = __get_image_name(folders, index, mri_type)

    return os.path.join(data_dir_path, folders[index], name)
