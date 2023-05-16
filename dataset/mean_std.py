import torch
from tqdm import tqdm
from load_paths import load_prepared_img_paths
import numpy as np
import config


def calc_mean_std(paths):
    mean_std_dict = {
        "t1": {"mean": 0, "std": 0},
        "t1ce": {"mean": 0, "std": 0},
        "t2": {"mean": 0, "std": 0},
        "flair": {"mean": 0, "std": 0},
    }

    modality = ["t1", "t1ce", "t2", "flair"]
    for pth in tqdm(paths, desc="Calcing means and stds"):
        for index, mod in enumerate(modality):
            data = np.load(pth)
            img = data["img"]
            img = torch.from_numpy(data["img"][index]).cuda().float()
            std, mean = torch.std_mean(img)
            mean_std_dict[mod]["mean"] += mean
            mean_std_dict[mod]["std"] += std

    for mod in modality:
        mean_std_dict[mod]["mean"] /= len(paths)
        mean_std_dict[mod]["std"] /= len(paths)

    return mean_std_dict


if __name__ == "__main__":
    print(calc_mean_std(load_prepared_img_paths(config.DATASET_PROCESSED_PATH)))


"""
{
    't1': {'mean': tensor(341.1601, device='cuda:0'), 'std': tensor(395.8610, device='cuda:0')}, 
    't1ce': {'mean': tensor(467.0914, device='cuda:0'), 'std': tensor(557.0898, device='cuda:0')},
    't2': {'mean': tensor(279.2286, device='cuda:0'), 'std': tensor(369.1520, device='cuda:0')},
    'flair': {'mean': tensor(219.8044, device='cuda:0'), 'std': tensor(270.5645, device='cuda:0')}}
"""
