import numpy as np
import pickle

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import tensorflow as tf
# from tensorflow.keras.models import *
from tensorflow.keras.saving import *
import os
import shap
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import mlflow
import json
from keras.models import load_model


def make_summary_plot(df, values, group, log_path, category='explainer'):
    plt.close()
    try:
        shap.summary_plot(values.values[:, :, 0], df, show=False)
    except:
        shap.summary_plot(values.values, df, show=False)
    f = plt.gcf()
    os.makedirs(f'{log_path}/shap/summary_{category}', exist_ok=True)
    plt.savefig(f'{log_path}/shap/summary_{category}/{group}_values.png')

    plt.close(f)


def make_force_plot(df, values, features, group, log_path,
                    category='explainer'):
    shap.force_plot(df, values, features=features, show=False)
    f = plt.gcf()
    os.makedirs(f'{log_path}/shap/force_{category}',
                exist_ok=True)
    plt.savefig(f'{log_path}/shap/force_{category}/{group}_values.png')
    mlflow.log_figure(f,
                      f'{log_path}/shap/force_{category}/{group}_values.png'
                      )

    plt.close(f)


def make_deep_beeswarm(df, values, group, log_path,
                       category='explainer'):
    shap.summary_plot(values, feature_names=df.columns,
                      features=df, show=False)
    f = plt.gcf()
    os.makedirs(f'{log_path}/shap/beeswarm_{category}',
                exist_ok=True)
    plt.savefig(f'{log_path}/shap/beeswarm_{category}/{group}_values.png')
    plt.close(f)


def make_decision_plot(df, values, misclassified, feature_names,
                       group, log_path, category='explainer'):
    shap.decision_plot(df, values, feature_names=list(feature_names),
                       show=False, link='logit', highlight=misclassified)
    f = plt.gcf()
    os.makedirs(f'{log_path}/shap/decision_{category}', exist_ok=True)
    plt.savefig(f'{log_path}/shap/decision_{category}/{group}_values.png')
    mlflow.log_figure(f,
                      f'{log_path}/shap/decision_{category}/{group}_values.png'
                      )
    plt.close(f)


def make_decision_deep(df, values, misclassified, feature_names,
                       group, log_path, category='explainer'):
    shap.decision_plot(df, values, feature_names=list(feature_names),
                       show=False, link='logit', highlight=misclassified)
    f = plt.gcf()
    os.makedirs(f'{log_path}/shap/decision_{category}',
                exist_ok=True
    )
    plt.savefig(
        f'{log_path}/shap/decision_{category}/{group}_values.png'
    )
    plt.close(f)

def make_beeswarm_plot(values, group, log_path, category='explainer'):
    shap.plots.beeswarm(values.values[:, :, 0], max_display=20, show=False)
    f = plt.gcf()
    os.makedirs(f'{log_path}/shap/beeswarm_{category}', exist_ok=True)
    plt.savefig(f'{log_path}/shap/beeswarm_{category}/{group}_values.png')
    plt.close(f)


def make_heatmap(values, group, log_path, category='explainer'):
    shap.plots.heatmap(values, instance_order=values.values.sum(1).argsort(), max_display=20, show=False)
    f = plt.gcf()
    os.makedirs(f'{log_path}/shap/heatmap_{category}', exist_ok=True)
    plt.savefig(f'{log_path}/shap/heatmap_{category}/{group}_values.png')
    plt.close(f)


def make_heatmap_deep(values, group, log_path, category='explainer'):
    shap.plots.heatmap(pd.DataFrame(values), instance_order=values.sum(1).argsort(), max_display=20, show=False)
    f = plt.gcf()
    os.makedirs(f'{log_path}/shap/heatmap_{category}', exist_ok=True)
    plt.savefig(f'{log_path}/shap/heatmap_{category}/{group}_values.png')
    plt.close(f)


def make_barplot(df, y, values, group, log_path, category='explainer'):
    clustering = shap.utils.hclust(df, y, metric='correlation')  # cluster_threshold=0.9
    # shap.plots.bar(values, max_display=20, show=False, clustering=clustering)
    shap.plots.bar(values.values, max_display=20, show=False,
                   clustering=clustering, clustering_cutoff=0.5)
    f = plt.gcf()
    os.makedirs(f'{log_path}/shap/bar_{category}', exist_ok=True)
    plt.savefig(f'{log_path}/shap/bar_{category}/{group}_values.png')
    plt.close(f)


def make_bar_plot(df, values, group, log_path, category='explainer'):
    shap.bar_plot(values, max_display=40, feature_names=df.columns, show=False)
    f = plt.gcf()
    os.makedirs(f'{log_path}/shap/barold_{category}', exist_ok=True)
    plt.savefig(f'{log_path}/shap/barold_{category}/{group}_values.png')
    plt.close(f)

def log_explainer(model, x_df, labels, group, log_path, v='all'):
    explainer = shap.Explainer(model)
    explanation = explainer(x_df)
    try:
        shap_values_df = pd.DataFrame(
            np.c_[explanation.base_values[:, 0], explanation.values[:, :, 0]], 
            columns=['bv'] + list(x_df.columns)
        )
    except:
        shap_values_df = pd.DataFrame(
            np.c_[explanation.base_values, explanation.values], 
            columns=['bv'] + list(x_df.columns)
        )
    shap_values_df = shap_values_df.abs()
    shap_values_df = shap_values_df.sum(0)
    total = shap_values_df.sum()
    shap_values_df = shap_values_df / total
    # Getting the base value
    bv = shap_values_df['bv']

    # Dropping the base value
    shap_values_df = shap_values_df.drop('bv')
    # make_images_shap(bins, shap_values_df, label, run, log_path=log_path)
    # sort the df
    shap_values_df = shap_values_df.sort_values(ascending=False)
    shap_values_df.to_csv(f"{log_path}/{group}_linear_shap_abs_{v}.csv")

    shap_values_df.transpose().hist(bins=100, figsize=(10, 10))
    plt.title(f'base_value: {np.round(bv, 2)}')
    plt.savefig(f"{log_path}/{group}_linear_shap_hist_abs_{v}.png")
    plt.close()

    # start x axis at 0
    shap_values_df.abs().sort_values(ascending=False).plot(kind='kde', figsize=(10, 10))
    # shap_values_df.transpose().cumsum().hist(bins=100, figsize=(10, 10))
    plt.xlim(0, shap_values_df.abs().max())
    plt.title(f'base_value: {np.round(bv, 2)}')
    plt.savefig(f"{log_path}/{group}_linear_shap_kde_abs_{v}.png")
    plt.close()

    values, base = np.histogram(shap_values_df.abs(), bins=40)
    #evaluate the cumulative
    cumulative = np.cumsum(values)
    # plot the cumulative function
    plt.plot(base[:-1], cumulative, c='blue')
    #plot the survival function
    plt.plot(base[:-1], len(shap_values_df.abs())-cumulative, c='green')

    plt.title(f'base_value: {np.round(bv, 2)}')
    plt.savefig(f"{log_path}/{group}_linear_shap_cumulative_abs_{v}.png")
    plt.close()


    if x_df.shape[1] <= 1000:
        # make_barplot(x_df, labels, explanation,
        #              log_path, 'test')

        # Summary plot
        make_summary_plot(x_df, explanation, 'test',
                          log_path, 'explainer')
        # make_beeswarm_plot(explanation, 'test',
        #                    log_path, 'explainer')
        # make_heatmap(explanation, 'test',
        #              log_path, 'explainer')
        # make_bar_plot(x_df, 'test',
        #               explanation.values[0], 'explainer')
        # make_force_plot(x_df, 'test', explanation.values[0],
        #                 x_df.columns, 'explainer')
    return shap_values_df

def log_shap(model, X, cols, labels, log_path, v=None):
    # explain all the predictions in the test set
    # explainer = shap.KernelExplainer(svc_linear.predict_proba, X_train[:100])
    os.makedirs(log_path, exist_ok=True)
    X_test_df = pd.DataFrame(X, columns=list(cols))
    shap_values_df = log_explainer(model, X_test_df, labels, "test", log_path, 'all')
    return shap_values_df

def make_feature_importance(args):
    ## Read data
    # root_path = "/home/melissa/Laval/Progressa/Journal/"
    features = pickle.load(open(f"{args.data_path}/features.pkl", "rb"))
    patients = pickle.load(open(f"{args.data_path}/patients.pkl", "rb"))
    labels = pickle.load(open(f"{args.data_path}/labels_2.pkl", "rb"))
    feature_names = pickle.load(open(f"{args.data_path}/feature_names.pkl", "rb"))

    indices_file = f"{args.data_path}/split_indices.csv"
    indices = np.loadtxt(open(indices_file), delimiter=",")

    path = f"{args.model}/{args.scaler}/{args.n_features}_features/n{args.n_splits}/endpoint{args.endpoint}/"
    if args.scaler == 'standard':
        scaler = StandardScaler()
    elif args.scaler == 'robust':
        scaler = RobustScaler()
    elif args.scaler == 'minmax':
        scaler = MinMaxScaler(feature_range=(-1, 1))
    else:
        scaler = None
    scaler.fit(features.reshape(-1, features.shape[-1]))
    print("SHAP importance")

    with tqdm(total=args.n_rep, position=0, leave=True) as pbar:
        shap_values = {i: None for i in range(args.n_rep)}
        shap_values_per_visit = {i: {v: None for v in range(6)} for i in range(args.n_rep)}
        for split_to_check in range(args.n_rep):
            train_indices = np.where(indices[:, split_to_check] == 1)[0]
            val_indices = np.where(indices[:, split_to_check] == 3)[0]

            X_train = features[train_indices]
            y_train = np.expand_dims(labels[train_indices], axis=-1)
            X_train = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)

            X_val = features[val_indices]
            y_val = np.expand_dims(labels[val_indices], axis=-1)
            # X_val = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(X_val.shape)
            X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
            X_val = np.nan_to_num(X_val, nan=-1)
            model = pickle.load(open(f"{args.results_path}/{path}/weights/{split_to_check}.pkl", 'rb'))

            ### SHAP ###
            os.makedirs(f"{args.results_path}/{path}/SHAP", exist_ok=True)
            df = pd.DataFrame(X_val.reshape(-1, len(feature_names)), columns=feature_names)
            is_row_all_neg = np.all(df == -1, axis=1)
            shap_values[split_to_check] = log_shap(model, df[~is_row_all_neg], feature_names,
                     y_val.reshape(-1)[~is_row_all_neg],
                     f"{args.results_path}/{path}/SHAP"
            )
            # all_grad_imp.append(gradient_importance(X_val, model, 0))
            for v in range(6):
                # if y_val[idx, 0] == 1:
                df = pd.DataFrame(X_val[:, v, :], columns=feature_names)
                is_row_all_neg = np.all(df == -1, axis=1)
                shap_values_per_visit[split_to_check][v] = log_shap(model, df[~is_row_all_neg], feature_names,
                        y_val[:,v].reshape(-1)[~is_row_all_neg], f"{args.results_path}/{path}/SHAP", v
                )

            # average all the shap values
            pbar.update(1)

    new_shap_values = {k: [] for k in shap_values[0].keys()}
    for k, _ in shap_values.items():
        if shap_values[k] is None:
            break
        for i, df in shap_values[k].items():
            new_shap_values[i].append(df)
    new_shap_values = {k: np.mean(v) for k, v in new_shap_values.items()}
    # Sort by values
    new_shap_values = {k: v for k, v in sorted(new_shap_values.items(), key=lambda item: item[1], reverse=True)}
    with open(f"{args.results_path}/{path}/SHAP/feature_importance.json", "w") as f:
        json.dump(new_shap_values, f)

    # TODO Il y a un probleme ici toutes les valeurs sont pareilles
    for v in range(6):
        new_shap_values = {k: [] for k in shap_values[0].keys()}
        for k, _ in shap_values_per_visit[v].items():
            if shap_values_per_visit[k][v] is None:
                break
            for i, df in shap_values_per_visit[k][v].items():
                new_shap_values[i].append(df)
        new_shap_values = {k: np.mean(v) for k, v in new_shap_values.items()}
        # Sort by values
        new_shap_values = {k: v for k, v in sorted(new_shap_values.items(), key=lambda item: item[1], reverse=True)}
        with open(f"{args.results_path}/{path}/SHAP/feature_importance_v{v}.json", "w") as f:
            json.dump(new_shap_values, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="xgboost")
    parser.add_argument("--data_path", type=str, default="data/")
    parser.add_argument("--results_path", type=str, default="results/")
    parser.add_argument("--n_rep", type=int, default=100)
    parser.add_argument("--n_features", type=int, default=-1)
    parser.add_argument("--scaler", type=str, default="minmax")
    parser.add_argument("--n_splits", type=int, default=100)
    parser.add_argument("--endpoint", type=int, default=2)
    parser.add_argument("--n_neurons", type=int, default=16)

    args = parser.parse_args()
    if args.model not in ['GRU', 'LSTM']:
        make_feature_importance(args)


if __name__ == "__main__":
    main()

