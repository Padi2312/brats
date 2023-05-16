import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from .load_paths import load_prepared_img_paths


class BratsDataset2D(Dataset):
    def __init__(self, data, custom_transforms=None, custom_transforms_label=None):
        self.custom_transforms = custom_transforms
        self.custom_transforms_label = custom_transforms_label
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data = self.__load_data(self.data[index])
        image = torch.from_numpy(data["img"])
        label = torch.from_numpy(data["label"])
        label = label.unsqueeze(0)

        if self.custom_transforms is not None:
            state = torch.get_rng_state()
            image = self.custom_transforms(image)
            torch.set_rng_state(state)
            if self.custom_transforms_label is not None:
                label = self.custom_transforms_label(label)

        label = label.squeeze(0)
        label = torch.nn.functional.one_hot(label.long(), 4)
        label = torch.movedim(label, -1, 0).to(torch.float)
        return image, label

    def __load_data(self, path):
        return np.load(path)


if __name__ == "__main__":
    data = load_prepared_img_paths("D:\\bratsdata_processed")

    custom_transforms = transforms.Compose(
        transforms=[
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            # transforms.RandomRotation(90),
        ]
    )

    dataset = BratsDataset2D(
        data=data,
    )

    x, y = dataset.__getitem__(102)

    y = torch.argmax(y, dim=0)

    # fig, ax = plt.subplots(ncols=2)
    # ax[0].imshow(x[1], cmap="gray")
    # ax[0].imshow(y, alpha=0.3, cmap="viridis")
