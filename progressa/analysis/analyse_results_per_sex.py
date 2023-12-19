import numpy as np
import os
import pickle
from sklearn import metrics
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import argparse

def analyse_results(data_path, results_path):
    patients = pickle.load(open(f"{data_path}/patients.pkl", "rb"))
    main_DB = pd.read_excel(f'{data_path}/2020_12_30_Databse_PROGRESSA.xlsx', sheet_name='Main Database')
    features = pickle.load(open(f"{data_path}/features.pkl", "rb"))
    scaler = MinMaxScaler(feature_range=(-1, 1))
    flag = 0
    for col in main_DB.columns:
        if flag == 0:
            header = main_DB[col].iloc[0]  # grab the first row for the header
            values = list(main_DB[col][1:])  # take the data less the header row
            main_DB_r = pd.DataFrame(data={header: values})
            flag = 1
        else:
            header = main_DB[col].iloc[0] #grab the first row for the header
            values = list(main_DB[col][1:]) #take the data less the header row
            main_DB_r[header] = values

    scores_names = ["#patients", "accuracy", "recall", "precision", "f-score", "phi-coefficient", "AUC"]
    for method in sorted(os.listdir(results_path)):
        method_path = results_path + "/" + method
        indices_file = f"{data_path}/split_indices.csv"
        indices = np.loadtxt(open(indices_file), delimiter=",")[:, 1:]
        if os.path.isdir(method_path):
            print(method)
            if method not in "RNN":
                continue
            predictions = pickle.load(open(method_path + "/predictions.pkl", "rb"))
            raw_predictions = pickle.load(open(method_path + "/predictions_raw.pkl", "rb"))

            real_values = pickle.load(open(method_path + "/real_values.pkl", "rb"))
            line_to_print = " ".join(scores_names)
            print("#visits "+line_to_print)
            # print("#visits", scores_names)
            for v in predictions.keys():
                if v < 0:
                    continue
                scores = np.zeros((10, len(scores_names)))

                for i in range(10):
                    these_test_indices = np.where(indices[:, i] == 2)[0]
                    these_train_indices = np.where(indices[:, i] == 1)[0]

                    X_train = features[these_train_indices]
                    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
                    X_train = np.nan_to_num(X_train, nan=-1)

                    X_test = features[these_test_indices]
                    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
                    X_test = np.nan_to_num(X_test, nan=-1)
                    these_real_values = []
                    these_predictions = []
                    these_raw_predictions = []
                    test_patients = patients[these_test_indices]
                    count = 0
                    for p_idx, patient in enumerate(test_patients):
                        if X_test[p_idx, v].max() != -1:
                            if main_DB_r[(main_DB_r["PSA ID"] == patient) & (main_DB_r["study_phase"] != "PO1y") & (main_DB_r["visit_number"] == v + 1)]["age"].item() > 69:
                            # if main_DB_r[(main_DB_r["PSA ID"] == patient) & (main_DB_r["study_phase"] != "PO1y") & (
                            #         main_DB_r["visit_number"] == v + 1)]["sex"].item() == 1:
                                these_real_values.append(real_values[v][i][count])
                                these_predictions.append(predictions[v][i][count])
                                these_raw_predictions.append(raw_predictions[v][i][count])
                            count += 1
                    if len(these_predictions) > 0:
                        scores[i][0] = len(these_predictions)
                        scores[i][1] = metrics.accuracy_score(these_real_values, these_predictions)
                        scores[i][2] = metrics.recall_score(these_real_values, these_predictions)
                        scores[i][3] = metrics.precision_score(these_real_values, these_predictions)
                        scores[i][4] = metrics.f1_score(these_real_values, these_predictions)
                        scores[i][5] = metrics.matthews_corrcoef(these_real_values, these_predictions)
                        fpr, tpr, thresholds = metrics.roc_curve(these_real_values, these_raw_predictions)
                        # fpr, tpr, thresholds = metrics.roc_curve(these_real_values, these_predictions)
                        scores[i][6] = metrics.auc(fpr, tpr)
                line_to_print = str(v + 1)
                for s in range(len(scores_names)):
                    line_to_print += " "+str(np.nanmean(scores[:, s]))
                    # line_to_print += " " + str(np.nanstd(scores[:, s]))
                print(line_to_print)
                line_to_print = str(v + 1)
                for s in range(len(scores_names)):
                    conf_interval = stats.bootstrap((scores[:, s],), np.mean, method="basic")
                    line_to_print += " " + str(conf_interval.confidence_interval)
                    # line_to_print += " " + str(np.nanstd(scores[:, s]))
                print(line_to_print)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, default="results/")
    parser.add_argument("--data_path", type=str, default="data/")

    args = parser.parse_args()

    analyse_results(args.data_path, args.results_path)


if __name__ == "__main__":
    main()
