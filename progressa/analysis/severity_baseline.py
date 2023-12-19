import pickle
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import argparse

def baseline(data_path):
    ## Read data
    features = pickle.load(open(f"{data_path}/features.pkl", "rb"))
    patients = pickle.load(open(f"{data_path}/patients.pkl", "rb"))
    labels = pickle.load(open(f"{data_path}/labels.pkl", "rb"))
    feature_names = pickle.load(open(f"{data_path}/feature_names.pkl", "rb"))

    id_as_severity = []
    for f_idx, f_n in enumerate(feature_names):
        if "AS_severity" or "AS severity" in f_n:
            id_as_severity.append(f_idx)
    indices_file = f"{data_path}/split_indices.csv"
    indices = np.loadtxt(open(indices_file), delimiter=",")[:, 1:]


    predictions = []
    real_values = []

    scaler = MinMaxScaler(feature_range=(-1, 1))

    for i in range(10):
        these_train_indices = np.where(indices[:, i] == 1)[0]
        these_test_indices = np.where(indices[:, i] == 2)[0]
        X_train = features[these_train_indices]
        y_train = np.expand_dims(labels[these_train_indices], axis=-1)
        X_train = np.nan_to_num(X_train, nan=-1)
        # X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        scaler.fit(X_train.reshape(-1, X_train.shape[-1])) # .reshape(X_train.shape)

        X_test = features[these_test_indices]
        y_test = np.expand_dims(labels[these_test_indices], axis=-1)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        X_test = np.nan_to_num(X_test, nan=-1)

        for p_idx in range(X_test.shape[0]):
            # print(id_as_severity)
            severities = []
            for v_id in range(X_test.shape[1]):
                if X_test[p_idx, v_id].max() > -1:

                    severities.append(X_test[p_idx, v_id, np.asarray(id_as_severity)].argmax())

            if severities[-1] > severities[0]:
                predictions.append(1)
            else:
                predictions.append(0)
            real_values.append(labels[p_idx, 0])

    scores_names = ["#patients", "accuracy", "sensitivity", "specificity", "f-score", "phi-coefficient", "AUC"]
    scores = np.zeros(len(scores_names))
    these_real_values = np.asarray(real_values)
    these_predictions = np.asarray(predictions)
    # print(these_real_values)
    # print(these_predictions)
    scores[0] = len(these_predictions)
    scores[1] = metrics.accuracy_score(these_real_values, these_predictions)
    scores[2] = metrics.recall_score(these_real_values, these_predictions)
    scores[3] = metrics.precision_score(these_real_values, these_predictions)
    scores[4] = metrics.f1_score(these_real_values, these_predictions)
    scores[5] = metrics.matthews_corrcoef(these_real_values, these_predictions)
    fpr, tpr, thresholds = metrics.roc_curve(these_real_values, np.asarray(predictions))
    scores[6] = metrics.auc(fpr, tpr)
    line_to_print = ""
    for s in range(len(scores_names)):
        line_to_print += " " + str(scores[s])
    print(line_to_print)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/")

    args = parser.parse_args()

    baseline(args.data_path)


if __name__ == "__main__":
    main()

