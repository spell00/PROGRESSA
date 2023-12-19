import pickle
import numpy as np
import os
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import calibration_curve
from matplotlib import pyplot as plt
import argparse

def make_calibration_plot(args):
    features = pickle.load(open(f"{args.data_path}/{args.features_file}", "rb"))
    labels = pickle.load(open(f"{args.data_path}/{args.labels_file}", "rb"))
    # patients = pickle.load(open(f"{args.data_path}/{args.patients_file}", "rb"))
    # feature_names = pickle.load(open("../feature_names.pkl", "rb"))

    indices_file = f"{args.data_path}/split_indices.csv"
    indices = np.loadtxt(open(indices_file), delimiter=",")[:, 1:]

    n_max_visits = features.shape[1]

    ### RNN
    rnn_output_path = "results/GRU/"
    if not os.path.exists(rnn_output_path):
        os.makedirs(rnn_output_path)

    rnn_weights_path = rnn_output_path + "weights/"
    if not os.path.exists(rnn_weights_path):
        os.makedirs(rnn_weights_path)

    predictions_values_per_visit = {}
    for v in range(n_max_visits):
        predictions_values_per_visit[v] = [[] for i in range(10)]
    predictions_values_per_visit[-1] = [[] for i in range(10)]

    real_values_per_visit = {}
    for v in range(n_max_visits):
        real_values_per_visit[v] = [[] for i in range(10)]
    real_values_per_visit[-1] = [[] for i in range(10)]

    predictions_raw_values_per_visit = {}
    for v in range(n_max_visits):
        predictions_raw_values_per_visit[v] = [[] for i in range(10)]
    predictions_raw_values_per_visit[-1] = [[] for i in range(10)]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    fpr_mean = np.linspace(0, 1, 100)
    interp_tprs = []
    # for i in range(100):
    #     interp_tpr = np.interp(fpr_mean, results['fpr'][i], results['tpr'][i])
    #     interp_tpr[0] = 0.0
    #     interp_tprs.append(interp_tpr)
    # tpr_mean = np.mean(interp_tprs, axis=0)
    # tpr_std = np.std(interp_tprs, axis=0)

    print("RNN")
    for i in range(args.n_rep):
        these_train_indices = np.where(indices[:, i] == 1)[0]
        these_test_indices = np.where(indices[:, i] == 2)[0]
        these_val_indices = np.where(indices[:, i] == 3)[0]

        X_train = features[these_train_indices]
        y_train = np.expand_dims(labels[these_train_indices], axis=-1)
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_train = np.nan_to_num(X_train, nan=-1)

        X_test = features[these_test_indices]
        y_test = np.expand_dims(labels[these_test_indices], axis=-1)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        X_test = np.nan_to_num(X_test, nan=-1)

        X_val = features[these_val_indices]
        y_val = np.expand_dims(labels[these_val_indices], axis=-1)
        X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_val = np.nan_to_num(X_val, nan=-1)

        input = Input(shape=(None, X_train.shape[-1]))
        lstm = Masking(mask_value=-1)(input)
        if args.model == 'GRU':
            lstm = GRU(16, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)(lstm)
        elif args.model == 'LSTM':
            lstm = LSTM(16, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)(lstm)
        else:
            exit('WRONG MODEL NAME')
        lstm = Dense(1, activation="sigmoid")(lstm)

        model = Model(input, lstm)
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        # Load the weights
        model.load_weights(f"{rnn_weights_path}/{i}.h5")

        predictions_raw = model.predict(X_test)[:, :, 0]
        y_test = y_test[:, :, 0]

        ## Calibration plot time

        # We flatten the predictions
        flat_prediction = predictions_raw.reshape((-1))
        flat_y_test = y_test.reshape((-1))

        prob_true, prob_pred = calibration_curve(flat_y_test, flat_prediction)
        interp_tprs += [np.interp(fpr_mean, prob_true, prob_pred)]
    tpr_mean = np.mean(np.stack(interp_tprs), axis=0)
    tpr_std = np.std(np.stack(interp_tprs), axis=0)
    tpr_upper = np.clip(tpr_mean + tpr_std, 0, 1)
    tpr_lower = tpr_mean - tpr_std
    plt.plot(fpr_mean, tpr_mean)
    plt.fill_between(fpr_mean, tpr_lower, tpr_upper, alpha=.2)

    plt.legend(loc="lower right")
    plt.xlabel("Mean predicted probability of positive class")
    plt.ylabel("Fraction of positives (Targets)")
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.savefig("Calibration_plot.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, default="results/")
    parser.add_argument("--features_file", type=str, default="features")
    parser.add_argument("--labels_file", type=str, default="labels")
    parser.add_argument("--n_features", type=int, default=-1)
    parser.add_argument("--data_path", type=str, default="data")
    # parser.add_argument("--indices_file", type=str, default="features/split_indices.csv")
    parser.add_argument("--model", type=str, default="GRU", help='choose one of [GRU, LSTM]')
    parser.add_argument("--n_rep", type=int, default=100)
    parser.add_argument("--endpoint", type=str, default='2', help="Final endpoint or 2 years ['2', 'final']")

    args = parser.parse_args()
    args.labels_file = f"{args.labels_file}_{args.endpoint}.pkl"
    if args.n_features == -1:
        args.features_file = f"{args.features_file}.pkl"
    else:
        args.features_file = f"{args.features_file}-{args.n_features}.pkl"
    make_calibration_plot(args)


if __name__ == "__main__":
    main()
