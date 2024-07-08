import numpy as np
import os
import pickle
from sklearn import metrics
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import argparse

def analyse_results(args):
    patients = pickle.load(open(f"{args.data_path}/patients.pkl", "rb"))
    main_DB = pd.read_excel(f'{args.data_path}/2020_12_30_Databse_PROGRESSA.xlsx', sheet_name='Main Database')
    features = pickle.load(open(f"{args.data_path}/features.pkl", "rb"))
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

    scores_names = ["#patients", "accuracy", "TPR", "TNR", "precision", "f-score", "phi-coefficient", "AUC"]
    path = f"{args.scaler}/{args.n_features}_features/n{args.n_splits}/endpoint{args.endpoint}/"
    for method in sorted(os.listdir(args.results_path)):
        method_path = f"{args.results_path}/{method}/{path}"
        if method in ["LSTM", "GRU"]:
            method_path = f"{method_path}/{args.n_neurons}"
        indices_file = f"{args.data_path}/split_indices.csv"
        indices = np.loadtxt(open(indices_file), delimiter=",")
        if os.path.isdir(method_path):
            print(method)
            predictions = pickle.load(open(method_path + f"/predictions.pkl", "rb"))
            raw_predictions = pickle.load(open(method_path + f"/predictions_raw.pkl", "rb"))
            real_values = pickle.load(open(method_path + f"/real_values.pkl", "rb"))

            line_to_print = " ".join(scores_names)
            print("#visits "+line_to_print)
            intervals_file = {x: open(f'{method_path}/intervals_{x}.tsv', 'w') for x in ['o69', 'u69', 'sex0', 'sex1', 'bi', 'tri', 'all']}
            avg_scores_files = {x: open(f'{method_path}/avg_scores_{x}.tsv', 'w') for x in ['o69', 'u69', 'sex0', 'sex1', 'bi', 'tri', 'all']}
            intervals_file['o69'].write("#visits\t"+"\t".join(line_to_print.split(" "))+"\n")
            avg_scores_files['o69'].write("#visits\t"+"\t".join(line_to_print.split(" "))+"\n")
            intervals_file['u69'].write("#visits\t"+"\t".join(line_to_print.split(" "))+"\n")
            avg_scores_files['u69'].write("#visits\t"+"\t".join(line_to_print.split(" "))+"\n")
            intervals_file['sex0'].write("#visits\t"+"\t".join(line_to_print.split(" "))+"\n")
            avg_scores_files['sex0'].write("#visits\t"+"\t".join(line_to_print.split(" "))+"\n")
            intervals_file['sex1'].write("#visits\t"+"\t".join(line_to_print.split(" "))+"\n")
            avg_scores_files['sex1'].write("#visits\t"+"\t".join(line_to_print.split(" "))+"\n")

            intervals_file['bi'].write("#visits\t"+"\t".join(line_to_print.split(" "))+"\n")
            avg_scores_files['bi'].write("#visits\t"+"\t".join(line_to_print.split(" "))+"\n")
            intervals_file['tri'].write("#visits\t"+"\t".join(line_to_print.split(" "))+"\n")
            avg_scores_files['tri'].write("#visits\t"+"\t".join(line_to_print.split(" "))+"\n")

            intervals_file['all'].write("#visits\t"+"\t".join(line_to_print.split(" "))+"\n")
            avg_scores_files['all'].write("#visits\t"+"\t".join(line_to_print.split(" "))+"\n")
            # print("#visits", scores_names)
            for v in predictions.keys():
                if v < 0:
                    continue
                zeros = np.zeros((len(predictions[v]), len(scores_names)))
                scores = {'o69': np.zeros((len(predictions[v]), len(scores_names))),
                          'u69': np.zeros((len(predictions[v]), len(scores_names))),
                          'sex0': np.zeros((len(predictions[v]), len(scores_names))),
                          'sex1': np.zeros((len(predictions[v]), len(scores_names))),
                          'bi': np.zeros((len(predictions[v]), len(scores_names))),
                          'tri': np.zeros((len(predictions[v]), len(scores_names))),
                          'all': np.zeros((len(predictions[v]), len(scores_names)))
                          }

                for i in range(len(predictions[v])):
                    these_test_indices = np.where(indices[:, i] == 2)[0]
                    these_train_indices = np.where(indices[:, i] == 1)[0]

                    X_train = features[these_train_indices]
                    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
                    X_train = np.nan_to_num(X_train, nan=-1)

                    X_test = features[these_test_indices]
                    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
                    X_test = np.nan_to_num(X_test, nan=-1)
                    these_real_values = {'o69': [], 'u69': [], 'sex0': [], 'sex1': [], 'bi': [], 'tri': [], 'all': []}
                    these_predictions = {'o69': [], 'u69': [], 'sex0': [], 'sex1': [], 'bi': [], 'tri': [], 'all': []}
                    these_raw_predictions = {'o69': [], 'u69': [], 'sex0': [], 'sex1': [], 'bi': [], 'tri': [], 'all': []}
                    test_patients = patients[these_test_indices]
                    count = 0
                    for p_idx, patient in enumerate(test_patients):
                        if X_test[p_idx, v].max() != -1:
                            sub_db = main_DB_r[(main_DB_r["PSA ID"] == patient) & (
                                # main_DB_r["study_phase"] != "PO1y") & (
                                    main_DB_r["visit_number"] == v + 1)]
                            if sub_db.shape[0] > 0:
                                if sub_db['age'].item() > 69:
                                    these_real_values['o69'].append(real_values[v][i][count])
                                    these_predictions['o69'].append(predictions[v][i][count])
                                    these_raw_predictions['o69'].append(raw_predictions[v][i][count])
                                else:
                                    these_real_values['u69'].append(real_values[v][i][count])
                                    these_predictions['u69'].append(predictions[v][i][count])
                                    these_raw_predictions['u69'].append(raw_predictions[v][i][count])

                                if sub_db['sex'].item() == 0:
                                    these_real_values['sex0'].append(real_values[v][i][count])
                                    these_predictions['sex0'].append(predictions[v][i][count])
                                    these_raw_predictions['sex0'].append(raw_predictions[v][i][count])
                                else:
                                    these_real_values['sex1'].append(real_values[v][i][count])
                                    these_predictions['sex1'].append(predictions[v][i][count])
                                    these_raw_predictions['sex1'].append(raw_predictions[v][i][count])
                                if sub_db['AO_valve_phenotype'].item() == 2.0:
                                    these_real_values['bi'].append(real_values[v][i][count])
                                    these_predictions['bi'].append(predictions[v][i][count])
                                    these_raw_predictions['bi'].append(raw_predictions[v][i][count])
                                elif sub_db['AO_valve_phenotype'].item() == 3.0:
                                    these_real_values['tri'].append(real_values[v][i][count])
                                    these_predictions['tri'].append(predictions[v][i][count])
                                    these_raw_predictions['tri'].append(raw_predictions[v][i][count])
                                these_real_values['all'].append(real_values[v][i][count])
                                these_predictions['all'].append(predictions[v][i][count])
                                these_raw_predictions['all'].append(raw_predictions[v][i][count])

                            count += 1
                    for k in these_predictions.keys():
                        if len(these_predictions[k]) > 0:
                            # these_predictions[k] = np.asarray(predictions[v][i])
                            scores[k][i][0] = len(these_predictions[k])
                            scores[k][i][1] = metrics.accuracy_score(these_real_values[k], these_predictions[k])
                            scores[k][i][2] = metrics.recall_score(these_real_values[k], these_predictions[k])
                            scores[k][i][3] = metrics.recall_score(these_real_values[k], these_predictions[k], pos_label=0)
                            scores[k][i][4] = metrics.precision_score(these_real_values[k], these_predictions[k])
                            scores[k][i][5] = metrics.f1_score(these_real_values[k], these_predictions[k])
                            scores[k][i][6] = metrics.matthews_corrcoef(these_real_values[k], these_predictions[k])
                            fpr, tpr, thresholds = metrics.roc_curve(these_real_values[k], np.asarray(these_raw_predictions[k]))
                            scores[k][i][7] = metrics.auc(fpr, tpr)
                        else:
                            pass

                for k in scores.keys():
                    intervals_to_print = str(v + 1)
                    avg_scores_to_print = str(v + 1)
                    for s in range(len(scores_names)):
                        conf_interval = stats.bootstrap((scores[k][:, s],), np.nanmean, method="basic")
                        intervals_to_print += "\t" + str(conf_interval.confidence_interval)
                        avg_scores_to_print += "\t" + str(np.nanmean(scores[k][:, s]))
                    intervals_file[k].write(intervals_to_print+"\n")
                    avg_scores_files[k].write(avg_scores_to_print+"\n")
            for k in intervals_file.keys():
                intervals_file[k].close()
                avg_scores_files[k].close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, default="results/")
    parser.add_argument("--data_path", type=str, default="data/")
    parser.add_argument("--n_features", type=int, default=-1)
    parser.add_argument("--endpoint", type=int, default=5)
    parser.add_argument("--n_splits", type=int, default=100)
    parser.add_argument("--n_neurons", type=int, default=16)
    parser.add_argument("--scaler", type=str, default='minmax')

    args = parser.parse_args()

    analyse_results(args)


if __name__ == "__main__":
    main()
