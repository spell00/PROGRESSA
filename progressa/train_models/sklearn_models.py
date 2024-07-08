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
        super().__init__(args)
        if args.n_visits == 2:
            self.get_data = utils.get_data_2v
        else:
            self.get_data = self.get_data

    def get_data(self, set_features, set_labels, indices):
        new_features = []
        new_labels = []
        patient_visit = []
        for p_idx in range(set_features.shape[0]):
            for v in range(set_features.shape[1]):
                if np.nan_to_num(set_features[p_idx, v], nan=-1).max() != -1:
                    new_features.append(set_features[p_idx, v])
                    new_labels.append(set_labels[p_idx, v])
                    patient_visit.append(np.array([indices[p_idx], v]))
                else:
                    pass
        set_features = np.asarray(new_features)
        set_labels = np.asarray(new_labels)
        patient_visit = np.asarray(patient_visit)

        return set_features, set_labels, patient_visit

    def train(self):
        for i in range(self.args.n_splits):
            self.make_indices_dict(i)
            # self.split_data(i)
            X_train = self.features[self.indices_dict['train']]
            y_train = self.labels[self.indices_dict['train']]
            X_test = self.features[self.indices_dict['test']]
            y_test = self.labels[self.indices_dict['test']]
            X_val = self.features[self.indices_dict['valid']]
            y_val = self.labels[self.indices_dict['valid']]

            X_train, y_train, patient_visit_train = self.get_data(X_train, y_train, self.indices_dict['train'])
            X_test, y_test, patient_visit_test = self.get_data(X_test, y_test, self.indices_dict['test'])
            X_val, y_val, patient_visit_val = self.get_data(X_val, y_val, self.indices_dict['valid'])

            if self.scaler is not None:
                X_train = self.scaler.transform(X_train)
                X_test = self.scaler.transform(X_test)
                X_val = self.scaler.transform(X_val)

            X_train = np.nan_to_num(X_train, nan=-1)
            X_test = np.nan_to_num(X_test, nan=-1)
            X_val = np.nan_to_num(X_val, nan=-1)
            features = np.nan_to_num(self.features, nan=-1)
            features = features.reshape(-1, features.shape[-1])
            # Remove rows with only -1
            mask = np.all(features == -1, axis=1)
            features = features[~mask]
            # Count all -1
            n_missing = np.sum(features == -1)
            
            model = self.model()
            model.fit(X_train, y_train)
            pickle.dump(model, open(f"results/{self.path}/weights/{i}.pkl", "wb"))

            predictions = model.predict(X_test)
            predictions_raw = model.predict_proba(X_test)[:, 1]
            for v in range(self.n_max_visits):
                v_ids = np.where(patient_visit_test[:, 1] == v)[0]

                if len(v_ids) > 0:
                    self.predictions_values_per_visit[v][i] += predictions[v_ids].tolist()
                    self.predictions_raw_values_per_visit[v][i] += predictions_raw[v_ids].tolist()
                    self.real_values_per_visit[v][i] += y_test[v_ids].tolist()

                    if v == 0:
                        for patients_idx in np.unique(patient_visit_test[:, 0]):
                            this_patient_samples = np.where(patient_visit_test[:, 0] == patients_idx)[0]
                            self.predictions_values_per_visit[-1][i].append(predictions[this_patient_samples[-1]])
                            self.predictions_raw_values_per_visit[-1][i].append(predictions_raw[this_patient_samples[-1]])
                            self.real_values_per_visit[-1][i].append(y_test[this_patient_samples[-1]])
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
    parser.add_argument("--model", type=str, default="Logistic_Regression", help='choose one of [Logistic_Regression, naiveBayes, lightgbm]')
    parser.add_argument("--n_splits", type=int, default=100)
    parser.add_argument("--n_visits", type=int, default=1)
    parser.add_argument("--endpoint", type=str, default='2', help="Final endpoint or 2 years ['2', 'final']")

    args = parser.parse_args()

    args.labels_file = f"{args.labels_file}_{args.endpoint}.pkl"

    os.makedirs(f'results/{args.model}', exist_ok=True)
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
