import os
import numpy as np


class Paths:
    def __init__(self, root_path):
        self.root_path = root_path
        self.file_list = os.listdir(root_path)

    def get_file_path_list(self, mri_type='t1'):
        path_list = []
        mask_list = []
        for index in range(len(self.file_list)):
            path_list.append(self.get_file_path(index, mri_type))
            mask_list.append(self.get_file_path(index, mask=True))
        return np.array(path_list), np.array(mask_list)

    def get_file_path_list_multi_channel(self, discard_channels=[]):
        mri_type_list = ['t1', 't1ce', 't2', 'flair']
        path_list = []
        mask_list = []
        for index in range(len(self.file_list)):
            channels = []
            for mri_type in mri_type_list:
                channels.append(self.get_file_path(index, mri_type))
            path_list.append(channels)
            mask_list.append(self.get_file_path(index, mask=True))
        return np.array(path_list), np.array(mask_list)

    def get_file_path(self, index: int, mri_type='t1', mask: bool = False):
        name = self.__get_label_image_name(index) if mask else self.__get_image_name(index, mri_type)
        return os.path.join(self.root_path, self.file_list[index], name)

    def __get_image_name(self, index: int, mri_type: str):
        return self.file_list[index] + '_' + mri_type + '.nii.gz'

    def __get_label_image_name(self, index: int):
        return self.file_list[index] + '_seg.nii.gz'
