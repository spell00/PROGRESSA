import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
import argparse

def create_splits(args):
    features = pickle.load(open(f"{args.data_path}/features.pkl", "rb"))
    labels = pickle.load(open(f"{args.data_path}/labels_{args.endpoint}.pkl", "rb"))

    n_samples = len(features)
    ## 1 = train, 2 = test, 3 = val
    all_indices = np.ones((n_samples, args.n_splits))
    # all_indices[:, 0] = np.arange(1, n_samples + 1)
    for i in range(args.n_splits):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=i)
        train_index, test_val_index = next(sss.split(np.zeros(n_samples), labels[:, 0]))
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=i)
        test_index, val_index = next(sss.split(np.zeros(len(test_val_index)), labels[test_val_index, 0]))

        all_indices[test_val_index[test_index], i] = 2
        all_indices[test_val_index[val_index], i] = 3

        assert np.intersect1d(train_index, test_val_index[test_index]).size == 0
        assert np.intersect1d(train_index, test_val_index[val_index]).size == 0
        assert np.intersect1d(test_val_index[test_index], test_val_index[val_index]).size == 0
    np.savetxt(f"{args.data_path}/split_indices.csv", all_indices, delimiter=",", fmt='%d')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/")
    parser.add_argument("--n_splits", type=int, default=100)
    parser.add_argument("--endpoint", type=str, default="2")

    args = parser.parse_args()

    create_splits(args)


if __name__ == "__main__":
    main()
