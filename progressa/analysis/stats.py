import pickle
import numpy as np
import argparse

def get_stats(data_path):

    features = pickle.load(open(f"{data_path}/features.pkl", "rb"))
    # patients = pickle.load(open(f"{data_path}/patients.pkl", "rb"))
    labels = pickle.load(open(f"{data_path}/labels.pkl", "rb"))
    # feature_names = pickle.load(open(f"{data_path}/feature_names.pkl", "rb"))

    features = np.nan_to_num(features, nan=-1)

    n_visits = features.shape[1]
    n_patients_n_visits = np.zeros(n_visits)
    n_patients_n_visits_positive = np.zeros(n_visits)
    n_patients_at_least_n_visits = np.zeros(n_visits)
    n_patients_at_least_n_visits_positive = np.zeros(n_visits)

    for p_idx in range(features.shape[0]):
        for v in range(features.shape[1]):
            if features[p_idx, v].max() != -1:
                n_patients_at_least_n_visits[v] += 1
                if labels[p_idx, v] == 1:
                    n_patients_at_least_n_visits_positive[v] += 1
            else:
                n_patients_n_visits[v - 1] += 1
                if labels[p_idx, v - 1] == 1:
                    n_patients_n_visits_positive[v - 1] += 1
                break
            if v == (features.shape[1] - 1):
                n_patients_n_visits[v] += 1
                if labels[p_idx, v] == 1:
                    n_patients_n_visits_positive[v] += 1

    print("Patients N visits")
    for v in range(n_visits):
        print(v + 1, n_patients_n_visits[v], n_patients_n_visits_positive[v])

    print("Patients at least N visits")
    for v in range(n_visits):
        print(v + 1, n_patients_at_least_n_visits[v], n_patients_at_least_n_visits_positive[v])


    n_patients_n_visits = np.zeros(n_visits)
    n_patients_n_visits_positive = np.zeros(n_visits)
    n_patients_at_least_n_visits = np.zeros(n_visits)
    n_patients_at_least_n_visits_positive = np.zeros(n_visits)
    indices_file = f"{data_path}/split_indices.csv"
    indices = np.loadtxt(open(indices_file), delimiter=",")[:, 1:]

    for i in range(10):
        these_test_indices = np.where(indices[:, i] == 2)[0]
        X_test = features[these_test_indices]
        y_test = labels[these_test_indices]
        for p_idx in range(X_test.shape[0]):
            for v in range(X_test.shape[1]):
                if X_test[p_idx, v].max() != -1:
                    n_patients_at_least_n_visits[v] += 1
                    if y_test[p_idx, v] == 1:
                        n_patients_at_least_n_visits_positive[v] += 1
                else:
                    n_patients_n_visits[v - 1] += 1
                    if y_test[p_idx, v - 1] == 1:
                        n_patients_n_visits_positive[v - 1] += 1
                    break
                if v == (X_test.shape[1] - 1):
                    n_patients_n_visits[v] += 1
                    if y_test[p_idx, v] == 1:
                        n_patients_n_visits_positive[v] += 1



    print("Patients N visits")
    for v in range(n_visits):
        print(v + 1, n_patients_n_visits[v], n_patients_n_visits_positive[v])

    print("Patients at least N visits")
    for v in range(n_visits):
        print(v + 1, n_patients_at_least_n_visits[v], n_patients_at_least_n_visits_positive[v])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/")

    args = parser.parse_args()

    get_stats(args.data_path)


if __name__ == "__main__":
    main()

