import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import metrics
from sklearn.metrics import roc_curve

import argparse


def plot_roc(args):

    fig, axs = plt.subplots(2, 3, figsize=(30, 30), dpi=110)
    lw = 2

    for method in sorted(os.listdir(args.results_path))[::-1]:
        method_path = args.results_path + "/" + method
        if os.path.isdir(method_path) & ("2v" not in method):
            print(method)
            if method not in ["Logistic_Regression", "GRU"]:
                continue
            predictions = pickle.load(open(method_path + f"/predictions_{args.endpoint}.pkl", "rb"))
            raw_predictions = pickle.load(open(method_path + f"/predictions_raw_{args.endpoint}.pkl", "rb"))

            real_values = pickle.load(open(method_path + f"/real_values_{args.endpoint}.pkl", "rb"))
            n_max_visits = len(predictions.keys()) - 1

            for v in range(n_max_visits):
                fpr = []
                tpr = []
                if len(real_values[v]) > 0:
                    for i in range(args.n_splits):
                        these_real_values = np.asarray(real_values[v][i])
                        these_predictions = np.asarray(raw_predictions[v][i])
                        # fpr, tpr, thresholds = roc_curve(these_real_values, np.asarray(raw_predictions[v]))
                        this_fpr, this_tpr, thresholds = roc_curve(these_real_values, these_predictions, drop_intermediate=False)

                        fpr += this_fpr.tolist()
                        tpr += this_tpr.tolist()
                tpr.sort()
                fpr.sort()
                if "GRU" in method:
                    optimal_idx = np.argmax(np.asarray(tpr) - np.asarray(fpr))
                    axs[int(v / 3), v % 3].scatter(fpr[optimal_idx], tpr[optimal_idx], color="black", zorder=3, s=10)
                axs[int(v / 3), v % 3].plot(fpr, tpr, lw=lw, label=method + " AUC = {:.3f}".format(metrics.auc(fpr, tpr)))
                if v == 0:
                    axs[int(v / 3), v % 3].set_title("Baseline")
                else:
                    axs[int(v / 3), v % 3].set_title("Year " + str(v))
                # axs[int(v / 3), v % 3].set_ylabel("Sensitivity")
                # axs[int(v / 3), v % 3].set_xlabel("1-Specificity \n")
                axs[int(v / 3), v % 3].legend(loc="lower right")



    # plt.legend(loc="lower right")
    # Set common labels
    fig.text(0.5, 0.04, '1-Specificity', ha='center', va='center')
    fig.text(0.08, 0.5, "Sensitivity", ha='center', va='center', rotation='vertical')
    plt.savefig(f'results/roc_{args.endpoint}.png')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, default="results/")
    parser.add_argument("--n_splits", type=int, default=100)
    parser.add_argument("--endpoint", type=str, default='2', help="Final endpoint or 2 years ['2', 'final']")

    args = parser.parse_args()

    plot_roc(args)


if __name__ == "__main__":
    main()
