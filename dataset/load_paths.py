import os

def load_prepared_img_paths(data_dir_path: str):
    paths = []
    folders = os.listdir(data_dir_path)
    for index in range(len(folders)):
        for item in os.listdir(os.path.join(data_dir_path, folders[index])):
            paths.append(os.path.join(data_dir_path, folders[index], item))
    return paths
