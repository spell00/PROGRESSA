import numpy as np
import pickle

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
# from tensorflow.keras.models import *
from tensorflow.keras.saving import *
import os
import argparse
from tqdm import tqdm

# tf.enable_eager_execution()


def gradient_importance(seq, model, idx_last):
    seq = tf.Variable(seq[np.newaxis, :, :], dtype=tf.float32)

    with tf.GradientTape() as tape:
        predictions = model(seq)

    grads = tape.gradient(predictions, seq)

    # grads = tf.reduce_mean(grads, axis=1).numpy()[0]
    grads = grads.numpy()[0, idx_last]

    return grads


def make_feature_importance(args):
    ## Read data
    # root_path = "/home/melissa/Laval/Progressa/Journal/"
    features = pickle.load(open(f"{args.data_path}/features.pkl", "rb"))
    patients = pickle.load(open(f"{args.data_path}/patients.pkl", "rb"))
    labels = pickle.load(open(f"{args.data_path}/labels_2.pkl", "rb"))
    feature_names = pickle.load(open(f"{args.data_path}/feature_names.pkl", "rb"))

    indices_file = f"{args.data_path}/split_indices.csv"
    indices = np.loadtxt(open(indices_file), delimiter=",")[:, 1:]

    scaler = MinMaxScaler(feature_range=(-1, 1))

    print("Gradient importance")

    visits_per_feature = {}
    importances_per_feature = {}

    for v in range(6):
        print(f"Visit {v+1}/6")
        output_path = f"{args.data_path}/feature_importance_per_visit/{v+1}v/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        all_grad_imp = []
        with tqdm(total=args.n_rep, position=0, leave=True) as pbar:
            for split_to_check in range(args.n_rep):
                train_indices = np.where(indices[:, split_to_check] == 1)[0]
                val_indices = np.where(indices[:, split_to_check] == 3)[0]

                X_train = features[train_indices]
                y_train = np.expand_dims(labels[train_indices], axis=-1)
                X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)

                X_val = features[val_indices]
                y_val = np.expand_dims(labels[val_indices], axis=-1)
                # X_val = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(X_val.shape)
                X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
                X_val = np.nan_to_num(X_val, nan=-1)
                model = load_model(f"{args.results_path}/{path}/{args.model}/weights/{split_to_check}.h5")

                ### GRADIENTS IMPORTANCE ###
                for idx in range(X_val.shape[0]):
                    # if y_val[idx, 0] == 1:
                    if X_val[idx, v].max() > -1:
                        all_grad_imp.append(gradient_importance(X_val[idx], model, v))
                pbar.update(1)

        all_grad_imp = np.asarray(all_grad_imp)
        avg_grad_imp = np.linalg.norm(all_grad_imp, axis=0)

        ids = avg_grad_imp.argsort()[::-1][:len(avg_grad_imp)]
        avgs = avg_grad_imp[ids]
        gradients = []
        for id in ids:
            gradients.append([id, avg_grad_imp[id], feature_names[id]])
        # pickle.dump(gradients,
        #             open(output_path + "gradients.pkl", "wb"))

        ids_values = np.zeros((len(gradients), 2))
        names_feats = []
        for idx in range(len(gradients)):
            ids_values[idx, 0] = gradients[idx][0]
            ids_values[idx, 1] = gradients[idx][1]
            names_feats.append(gradients[idx][2])

        n_feats = 25
        # pickle.dump(features[:, :, np.asarray(ids_values[:25, 0], dtype=int)],
        #             open(output_path + "features-"+str(n_feats)+"pkl", "wb"))

        these_feature_names = [feature_names[int(id_v)] for id_v in ids_values[:n_feats, 0]]
        for f_n in these_feature_names:
            if f_n not in visits_per_feature.keys():
                visits_per_feature[f_n] = np.zeros(6)
                importances_per_feature[f_n] = np.zeros(6)
            visits_per_feature[f_n][v] = 1
            for gr_id, gr_val, gr_fn in gradients:
                if f_n == gr_fn:
                    importances_per_feature[f_n][v] = gr_val
                    break
    df = pd.DataFrame(columns=["v1", "v2", "v3", "v4", "v5", "v6"])
    for f_n in visits_per_feature.keys():
        row = []
        for i in range(6):
            row += [str(importances_per_feature[f_n][i] / importances_per_feature[f_n].sum())]
        # line_to_print += " " + str(importances_per_feature[f_n].mean()) + " " + str(importances_per_feature[f_n].std())
        df = pd.concat((df, pd.DataFrame(row, index=df.columns, columns=[f_n]).transpose()))
    df.to_csv("feature_importance.csv", index=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="GRU")
    parser.add_argument("--data_path", type=str, default="data/")
    parser.add_argument("--results_path", type=str, default="results/")
    parser.add_argument("--n_rep", type=int, default=100)

    args = parser.parse_args()

    make_feature_importance(args)


if __name__ == "__main__":
    main()

