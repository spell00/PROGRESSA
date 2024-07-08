import numpy as np
import os
import pickle
from sklearn import metrics
from scipy import stats
import argparse

def analyse_results(args):
    scores_names = ["#patients", "accuracy", "TPR", "TNR", "precision", "f-score", "phi-coefficient", "AUC"]
    path = f"{args.scaler}/{args.n_features}_features/n{args.n_splits}/endpoint{args.endpoint}/"
    for method in sorted(os.listdir(args.results_path)):
        method_path = f"{args.results_path}/{method}/{path}"
        if method in ["LSTM", "GRU"]:
            method_path = f"{method_path}/{args.n_neurons}"
        if "allvisits" in method:
            continue
        if os.path.isdir(method_path):
            print(method)
            predictions = pickle.load(open(method_path + f"/predictions.pkl", "rb"))
            raw_predictions = pickle.load(open(method_path + f"/predictions_raw.pkl", "rb"))
            real_values = pickle.load(open(method_path + f"/real_values.pkl", "rb"))
            line_to_print = " ".join(scores_names)
            intervals_file = open(f'{method_path}/intervals.tsv', 'w')
            avg_scores_files = open(f'{method_path}/avg_scores.tsv', 'w')
            intervals_file.write("#visits\t"+"\t".join(line_to_print.split(" "))+"\n")
            avg_scores_files.write("#visits\t"+"\t".join(line_to_print.split(" "))+"\n")

            for v in predictions.keys():
                scores = np.zeros((len(predictions[v]), len(scores_names)))
                for i in range(len(predictions[v])):
                    if len(real_values[v][i]) > 0:
                        these_real_values = np.asarray(real_values[v][i])
                        these_predictions = np.asarray(predictions[v][i])
                        scores[i][0] = len(these_predictions)
                        scores[i][1] = metrics.accuracy_score(these_real_values, these_predictions)
                        scores[i][2] = metrics.recall_score(these_real_values, these_predictions)
                        scores[i][3] = metrics.recall_score(these_real_values, these_predictions, pos_label=0)
                        scores[i][4] = metrics.precision_score(these_real_values, these_predictions)
                        scores[i][5] = metrics.f1_score(these_real_values, these_predictions)
                        scores[i][6] = metrics.matthews_corrcoef(these_real_values, these_predictions)
                        fpr, tpr, thresholds = metrics.roc_curve(these_real_values, np.asarray(raw_predictions[v][i]))
                        scores[i][7] = metrics.auc(fpr, tpr)
                    else:
                        pass
                intervals_to_print = str(v + 1)
                avg_scores_to_print = str(v + 1)
                for s in range(len(scores_names)):
                    conf_interval = stats.bootstrap((scores[:, s],), np.mean, method="basic")
                    intervals_to_print += "\t" + str(conf_interval.confidence_interval)
                    avg_scores_to_print += "\t" + str(np.nanmean(scores[:, s]))
                intervals_file.write(intervals_to_print+"\n")
                avg_scores_files.write(avg_scores_to_print+"\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_features", type=int, default=-1)
    parser.add_argument("--results_path", type=str, default="results/")
    parser.add_argument("--endpoint", type=int, default=2)
    parser.add_argument("--n_splits", type=int, default=100)
    parser.add_argument("--scaler", type=str, default='minmax')
    parser.add_argument("--n_neurons", type=int, default=16)

    args = parser.parse_args()

    analyse_results(args)


if __name__ == "__main__":
    main()
