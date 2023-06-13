import torch
from tqdm import tqdm
import numpy as np
from tqdm.contrib.concurrent import process_map


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

# if __name__ == "__main__":
#     mean_std = calc_mean_std(load_prepared_img_paths(config.DATASET_PROCESSED_PATH))
#     print(mean_std)
#     json.dump(mean_std)

