import numpy as np
import os
import pickle
import pandas as pd
from sklearn import metrics
from scipy import stats
import argparse

def analyse_results(args):
    scores_names = ["#patients", "accuracy", "recall", "precision", "f-score", "phi-coefficient", "AUC"]
    for method in sorted(os.listdir(args.results_path)):
        method_path = args.results_path + "/" + method
        if "allvisits" in method:
            continue
        if os.path.isdir(method_path):
            print(method)
            predictions = pickle.load(open(method_path + f"/predictions_{args.endpoint}_{args.n_features}.pkl", "rb"))
            raw_predictions = pickle.load(open(method_path + f"/predictions_raw_{args.endpoint}_{args.n_features}.pkl", "rb"))
            real_values = pickle.load(open(method_path + f"/real_values_{args.endpoint}_{args.n_features}.pkl", "rb"))
            scores = np.zeros((100, len(scores_names)))
            line_to_print = " ".join(scores_names)
            print("#visits "+line_to_print)
            # print("#visits", scores_names)
            for v in predictions.keys():
                scores = np.zeros((100, len(scores_names)))
                for i in range(100):
                    if len(real_values[v][i]) > 0:
                        these_real_values = np.asarray(real_values[v][i])
                        these_predictions = np.asarray(predictions[v][i])
                        scores[i][0] = len(these_predictions)
                        scores[i][1] = metrics.accuracy_score(these_real_values, these_predictions)
                        scores[i][2] = metrics.recall_score(these_real_values, these_predictions)
                        scores[i][3] = metrics.precision_score(these_real_values, these_predictions)
                        scores[i][4] = metrics.f1_score(these_real_values, these_predictions)
                        scores[i][5] = metrics.matthews_corrcoef(these_real_values, these_predictions)
                        fpr, tpr, thresholds = metrics.roc_curve(these_real_values, np.asarray(raw_predictions[v][i]))
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
    parser.add_argument("--n_features", type=int, default=-1)
    parser.add_argument("--results_path", type=str, default="results/")
    parser.add_argument("--endpoint", type=int)

    args = parser.parse_args()

    analyse_results(args)


if __name__ == "__main__":
    main()
