from dataset import calc_mean_std, load_prepared_img_paths
import json
import config

if __name__ == "__main__":
    data = load_prepared_img_paths(config.DATASET_PROCESSED_PATH)
    data = data[0 : config.NUM_IMGS]
    mean_std = calc_mean_std(data)
    print(mean_std)
    json.dump(mean_std)
