import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from sklearn import metrics
from sklearn.metrics import roc_curve

import argparse


def plot_roc(args):

    fig, axs = plt.subplots(2, 3, figsize=(30, 30), dpi=110)
    plt.rc('font', size=22)  # controls default text sizes
    plt.rc('figure', titlesize=22)  # fontsize of the figure title
    for method in sorted(os.listdir(args.results_path))[::-1]:
        print(method)
        if 'cumul' in method:
            cumul = 1
        else:
            cumul = 0
        if cumul != args.cumul and method not in ["GRU"]:
            continue
        if args.n_features in ['-1', '22']:
            path = f"{args.scaler}/{args.n_features}_features/n{args.n_splits}/endpoint{args.endpoint}/"
        else:
            if "Logistic_Regression" in method:
                path = f"{args.scaler}/22_features/n{args.n_splits}/endpoint{args.endpoint}/"
            else:
                path = f"{args.scaler}/-1_features/n{args.n_splits}/endpoint{args.endpoint}/"
        method_path = f"{args.results_path}/{method}/{path}"
        if "LSTM" in method or "GRU" in method:
            method_path = f"{method_path}/{args.n_neurons}"
        if os.path.isdir(method_path) & ("2v" not in method):
            if cumul == 1 and method not in ["GRU", "LSTM"]:
                method = "_".join(method.split("_")[:-1])
            if method not in ["Logistic_Regression", "GRU", "lightgbm"] and args.all_models == 0:
                continue
            if method == "Logistic_Regression":
                method = "LogReg"
            predictions = pickle.load(open(method_path + f"/predictions.pkl", "rb"))
            raw_predictions = pickle.load(open(method_path + f"/predictions_raw.pkl", "rb"))

            real_values = pickle.load(open(method_path + f"/real_values.pkl", "rb"))
            n_max_visits = len(predictions.keys()) - 1

            for v in range(n_max_visits):
                fpr = []
                tpr = []
                auc = []
                if len(real_values[v]) > 0:
                    for i in range(args.n_splits):
                        these_real_values = np.asarray(real_values[v][i])
                        these_predictions = np.asarray(raw_predictions[v][i])

                        # fpr, tpr, thresholds = roc_curve(these_real_values, np.asarray(raw_predictions[v]))
                        this_fpr, this_tpr, thresholds = roc_curve(these_real_values, these_predictions, drop_intermediate=False)

                        fpr += [this_fpr.tolist()]
                        tpr += [this_tpr.tolist()]
                        auc += [metrics.auc(fpr[-1], tpr[-1])]
                axs = plot_intervals({'fpr': fpr, 'tpr': tpr, 'auc': auc}, axs, v, method, args.n_splits)

    fig.text(0.5, 0.04, '1-Specificity', ha='center', va='center')
    fig.text(0.08, 0.5, "Sensitivity", ha='center', va='center', rotation='vertical')
    os.makedirs(f"results/roc/cumul{args.cumul}", exist_ok=True)
    plt.savefig(f'results/roc/cumul{args.cumul}/roc_{args.endpoint}_{args.n_features}_{args.scaler}_n{args.n_splits}_all{args.all_models}.png')


def plot_intervals(results, axs, v, method, n_splits):
    fpr_mean = np.linspace(0, 1, 100)
    interp_tprs = []
    for i in range(n_splits):
        interp_tpr = np.interp(fpr_mean, results['fpr'][i], results['tpr'][i])
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)

    tpr_mean = np.mean(interp_tprs, axis=0)
    tpr_std = np.std(interp_tprs, axis=0)

    # if "GRU" in method:
    #     optimal_idx = np.argmax(np.asarray(tpr_mean) - np.asarray(fpr_mean))
    #     axs[int(v / 3), v % 3].scatter(fpr_mean[optimal_idx], tpr_mean[optimal_idx], color="black", zorder=3, s=10)
    if v == 0:
        axs[int(v / 3), v % 3].set_title("Baseline")
    else:
        axs[int(v / 3), v % 3].set_title("Year " + str(v))
    for label in (axs[int(v / 3), v % 3].get_xticklabels() + axs[int(v / 3), v % 3].get_yticklabels()):
        # label.set_fontproperties(font_prop)
        label.set_fontsize(20)  # Size here overrides font_prop
    tpr_upper = np.clip(tpr_mean + tpr_std, 0, 1)
    tpr_lower = tpr_mean - tpr_std
    axs[int(v / 3), v % 3].plot(fpr_mean, tpr_mean, lw=2, label=f"{method} AUC = {np.round(np.mean(results['auc']), 2)} ± {np.round(np.std(results['auc']), 2)}")
    axs[int(v / 3), v % 3].fill_between(fpr_mean, tpr_lower, tpr_upper, alpha=.2)
    axs[int(v / 3), v % 3].legend(loc="lower right")

    return axs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, default="results/")
    parser.add_argument("--n_splits", type=int, default=100)
    parser.add_argument("--n_features", type=str, default='-1')
    parser.add_argument("--endpoint", type=str, default='2', help="Final endpoint or 2 years ['2', 'final']")
    parser.add_argument("--scaler", type=str, default='minmax')
    parser.add_argument("--cumul", type=int, default=0)
    parser.add_argument("--n_neurons", type=int, default=16)
    parser.add_argument("--all_models", type=int, default=0)

    args = parser.parse_args()

    plot_roc(args)


if __name__ == "__main__":
    main()
