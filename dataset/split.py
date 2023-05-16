from .dataset import BratsDataset2D
import random


def get_train_val_test_dataset(
    data,
    transforms=None,
    label_transforms=None,
    test_transforms=None,
    seed=2312,
    split={"train": 0.7, "val": 0.15},
):
    total_count = len(data)
    train_count = int(split["train"] * total_count)
    val_count = int(split["val"] * total_count)
    # test_count = total_count - train_count - val_count

    # Set seed for always getting the same datasets
    random.seed(seed)
    random.shuffle(data)

    train_data = data[:train_count]
    val_data = data[train_count : train_count + val_count]
    test_data = data[train_count + val_count :]

    train_dataset = BratsDataset2D(train_data, transforms, label_transforms)
    val_dataset = BratsDataset2D(val_data, transforms, label_transforms)

    # Don't apply custom transformations on test dataset
    test_dataset = BratsDataset2D(test_data, test_transforms, label_transforms)

    return train_dataset, val_dataset, test_dataset
