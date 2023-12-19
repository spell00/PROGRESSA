import kmapper as km
import pickle
from progressa import utils
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
import matplotlib.pyplot as plt

import seaborn as sns

from tensorflow.keras.models import *
from sklearn.preprocessing import MinMaxScaler
import argparse

def make_tsne(data_path, results_path):
    indices_file = f"{data_path}/split_indices.csv"
    indices = np.loadtxt(open(indices_file), delimiter=",")[:, 1:]
    features = pickle.load(open(f"{data_path}/features.pkl", "rb"))
    # features = pickle.load(open("preprocess/features-22.pkl", "rb"))
    labels = pickle.load(open(f"{data_path}/labels.pkl", "rb"))

    scaler = MinMaxScaler(feature_range=(-1, 1))


    tsne_features = []
    tsne_labels = []

    for split_to_check in range(10):

        train_indices = np.where(indices[:, split_to_check] == 1)[0]
        test_indices = np.where(indices[:, split_to_check] == 2)[0]

        X_train = features[train_indices]
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)

        X_test_raw = features[test_indices]
        X_test = features[test_indices]
        y_test = labels[test_indices]
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        X_test = np.nan_to_num(X_test, nan=-1)

        model = load_model(f"{results_path}/GRU/weights/{split_to_check}.h5")
        model = Model(inputs=model.input, outputs=model.layers[-2].output)

        X_test = model.predict(X_test)

        for p_idx in range(X_test_raw.shape[0]):
            for v in range(X_test_raw.shape[1]):
                if np.nan_to_num(X_test_raw[p_idx, v], nan=-1).max() == -1:
                    tsne_features.append(X_test[p_idx, v - 1])
                    tsne_labels.append(v - 1)
                    # new_labels.append(y_train[p_idx, v])
                    # tsne_labels.append(y_test[p_idx, v - 1])
                    break
                if v == (X_test_raw.shape[1] - 1):
                    tsne_features.append(X_test[p_idx, v])
                    # tsne_labels.append(y_test[p_idx, v])
                    tsne_labels.append(v)
    tsne_features = np.asarray(tsne_features)
    tsne_labels = np.asarray(tsne_labels)
    # print(X_train.shape, y_train.shape)

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(tsne_features)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    df_subset = {}
    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue=tsne_labels,
        palette=sns.color_palette("hls", np.max(tsne_labels)),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    plt.savefig('results/GRU/tsne.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/")
    parser.add_argument("--results_path", type=str, default="results/")

    args = parser.parse_args()

    make_tsne(args.data_path, args.results_path)


if __name__ == "__main__":
    main()
