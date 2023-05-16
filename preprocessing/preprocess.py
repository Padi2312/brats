import argparse
import torch
from load_paths import load_raw_data_paths
import torchvision.transforms as transforms
from tqdm import tqdm
from utils import get_tensor_from_nifti
import os
import torchvision.utils as vutils
import numpy as np
from multiprocessing.pool import Pool
from concurrent.futures import ThreadPoolExecutor
from functools import partial


class Preprocessing:
    def __init__(
        self, input_dir: str, output_dir: str, dimension: str, size: int = 128
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.dimension = dimension
        self.size = size

        if dimension == "2d":
            self.preprocess_2d()
            pass
        elif dimension == "3d":
            pass
        else:
            print("[ERROR] Parameter <dimension> should be '2d' or '3d'")
            pass

    def preprocess_2d(self):
        img_paths, mask_paths = load_raw_data_paths(self.input_dir)
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Create a list of futures for each pair of image and mask paths
            futures = [
                executor.submit(partial(self.__process_pair, channel_paths, seg_path))
                for channel_paths, seg_path in zip(img_paths, mask_paths)
            ]

            # Wait for all the futures to complete
            for future in futures:
                future.result()

    def __process_pair(self, channel_paths, seg_path):
        channels = {
            "t1": None,
            "t1ce": None,
            "t2": None,
            "flair": None,
            "seg": None,
        }

        # Process every channel of the image
        # Iterate over each channel in img_paths
        for single_img_path in channel_paths:
            channels[single_img_path] = self.__preprocess_image(
                get_tensor_from_nifti(channel_paths[single_img_path])
            )

        # Process the mask image
        y = self.__preprocess_label(get_tensor_from_nifti(seg_path))
        channels["seg"] = y
        self.__save_img_slices(path=seg_path, data=channels)
        pass

    def __resize_image(self, img):
        return transforms.Compose(
            [
                transforms.CenterCrop((self.size + 60, self.size + 60)),
                transforms.Resize(
                    (self.size, self.size), transforms.InterpolationMode.NEAREST
                ),
            ]
        )(img)

    def __reshape_slices(self, img):
        return img[14:142, ...]

    def __preprocess_image(self, img: torch.Tensor):
        processed_image = img.to(torch.float).cuda()
        processed_image = self.__reshape_slices(processed_image)
        processed_image = self.__resize_image(processed_image)
        processed_image = processed_image.to(torch.float)
        return processed_image

    def __preprocess_label(self, label: torch.Tensor):
        new_mask = label.cuda()
        new_mask = self.__reshape_slices(new_mask)
        new_mask = self.__resize_image(new_mask)
        new_mask[new_mask == 4] = 3
        new_mask = new_mask.to(torch.uint8)
        return new_mask

    def __save_img_slices(self, data, path):
        for slice_index in range(data["seg"].shape[0]):
            cimage = torch.zeros(4, self.size, self.size)
            label = None
            for index, ctype in enumerate(data):
                if ctype != "seg":
                    cimage[index] = data[ctype][slice_index]
                else:
                    label = data[ctype][slice_index]

            folder_path = os.path.join(self.output_dir, path.split(os.sep)[2])
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            save_path = os.path.join(
                self.output_dir,
                path.split(os.sep)[2],
                path.split("\\")[3].split(".")[0].split("_seg")[0]
                + f"_{slice_index}.npz",
            )
            np.savez_compressed(save_path, img=cimage.cpu(), label=label.cpu())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=str, default="2d")
    parser.add_argument("--input_dir", type=str, default="C:\\bratsdata")
    parser.add_argument("--output_dir", type=str, default="D:\\bratsdata_processed_128_cc")
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()

    print(args.dimension)
    print(args.input_dir)
    print(args.output_dir)

    Preprocessing(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dimension=args.dimension,
        size=args.size,
    )
