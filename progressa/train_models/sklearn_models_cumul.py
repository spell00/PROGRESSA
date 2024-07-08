import pickle
import numpy as np
import os
from ML import Classifier
import argparse
from progressa import utils
import pandas as pd


class SklearnClassifier(Classifier):
    """
    SklearnClassifier class
    """
    def __init__(self, args):
        """
        SklearnClassifier constructor
        """
        args.model = f"{args.model}_cumul"
        super().__init__(args)

    def train(self):
        for i in range(self.args.n_splits):
            self.make_indices_dict(i)
            # self.split_data(i)
            X_train = self.features[self.indices_dict['train']]
            Y_train = self.labels[self.indices_dict['train']]
            X_test = self.features[self.indices_dict['test']]
            Y_test = self.labels[self.indices_dict['test']]
            X_val = self.features[self.indices_dict['valid']]
            Y_val = self.labels[self.indices_dict['valid']]

            if self.scaler is not None:
                X_train = self.scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
                X_test = self.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
                X_val = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

            X_train = np.nan_to_num(X_train, nan=-1)
            X_test = np.nan_to_num(X_test, nan=-1)
            X_val = np.nan_to_num(X_val, nan=-1)

            for v in range(self.n_max_visits):
                to_keep = X_train[:, v].max(1) != -1
                x_train = X_train[to_keep, :v+1, :]
                x_train = x_train.reshape(x_train.shape[0], -1)
                y_train = Y_train[to_keep, v]

                to_keep = X_test[:, v].max(1) != -1
                x_test = X_test[to_keep, :v+1, :]
                x_test = x_test.reshape(x_test.shape[0], -1)
                y_test = Y_test[to_keep, v]

                to_keep = X_val[:, v].max(1) != -1
                x_val = X_val[to_keep, :v+1, :]
                x_val = x_val.reshape(x_val.shape[0], -1)
                y_val = Y_val[to_keep, v]

                model = self.model()
                model.fit(x_train, y_train)
                pickle.dump(model, open(f"results/{self.path}/weights/{i}.pkl", "wb"))

                predictions = model.predict(x_test)
                predictions_raw = model.predict_proba(x_test)[:, 1]
                self.predictions_values_per_visit[v][i] = predictions.tolist()
                self.predictions_raw_values_per_visit[v][i] = predictions_raw.tolist()
                self.real_values_per_visit[v][i] = y_test.tolist()
                # print(f'V{v}: {np.sum([x==y for x,y in zip(self.predictions_values_per_visit[v][i],self.real_values_per_visit[v][i])])/len(self.predictions_values_per_visit[v][i])}')
        pickle.dump(self.predictions_values_per_visit, open(f"results/{self.path}/predictions.pkl", "wb"))
        pickle.dump(self.predictions_raw_values_per_visit, open(f"results/{self.path}/predictions_raw.pkl", "wb"))
        pickle.dump(self.real_values_per_visit, open(f"results/{self.path}/real_values.pkl", "wb"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_features", type=int, default=-1)
    parser.add_argument("--features_file", type=str, default="features")
    parser.add_argument("--labels_file", type=str, default="labels")
    parser.add_argument("--patients_file", type=str, default="patients.pkl")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--scaler", type=str, default="minmax")

    # parser.add_argument("--indices_file", type=str, default="features/split_indices.csv")
    parser.add_argument("--model", type=str, default="Logistic_Regression", help='choose one of [Logistic_Regression, naiveBayes, lightgbm]')
    parser.add_argument("--n_splits", type=int, default=100)
    parser.add_argument("--n_visits", type=int, default=1)
    parser.add_argument("--endpoint", type=str, default='2', help="Final endpoint or 2 years ['2', 'final']")

    args = parser.parse_args()

    args.labels_file = f"{args.labels_file}_{args.endpoint}.pkl"

    os.makedirs(f'results/{args.model}_cumul', exist_ok=True)
    if args.n_features == -1:
        args.features_file = f"{args.features_file}.pkl"
    else:
        args.features_file = f"{args.features_file}-{args.n_features}.pkl"

    classifier = SklearnClassifier(args)
    classifier.load_data(
        f"{args.data_path}/{args.features_file}",
        f"{args.data_path}/{args.labels_file}",
        f"{args.data_path}/{args.patients_file}"
    )
    classifier.train()


if __name__ == "__main__":
    main()
