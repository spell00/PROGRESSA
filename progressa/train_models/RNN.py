import pickle
import numpy as np
import os
from ML import Classifier
import argparse
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.utils import set_random_seed
import tensorflow as tf
def get_data(set_features, set_labels, indices):
    new_features = []
    new_labels = []
    patient_visit = []
    for p_idx in range(set_features.shape[0]):
        for v in range(set_features.shape[1]):
            if np.nan_to_num(set_features[p_idx, v], nan=-1).max() != -1 or np.nan_to_num(set_labels[p_idx, v], nan=-1).max() != -1:
                new_features.append(set_features[p_idx, v])
                new_labels.append(set_labels[p_idx, v])
                patient_visit.append(np.array([indices[p_idx], v]))
            else:
                pass
    set_features = np.asarray(new_features)
    set_labels = np.asarray(new_labels)
    patient_visit = np.asarray(patient_visit)

    return set_features, set_labels, patient_visit

set_random_seed(seed=42)

class RNN(Classifier):
    """
    RNN class
    """
    def __init__(self, args):
        """
        RNN constructor
        """
        super().__init__(args)
        os.makedirs(f"results/{self.path}/{args.n_neurons}/weights/", exist_ok=True)

    def train(self):
        best_epoch = []
        for i in range(self.args.n_splits):
            print(f"Repetition: {i + 1}/{self.args.n_splits}")
            self.make_indices_dict(i)
            # self.split_data(i)
            X_train = self.features[self.indices_dict['train']]
            y_train = np.expand_dims(self.labels[self.indices_dict['train']], axis=-1)
            X_test = self.features[self.indices_dict['test']]
            y_test = np.expand_dims(self.labels[self.indices_dict['test']], axis=-1)
            X_val = self.features[self.indices_dict['valid']]
            y_val = np.expand_dims(self.labels[self.indices_dict['valid']], axis=-1)

            if self.scaler is not None:
                X_train = self.scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
                X_test = self.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
                X_val = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

            X_train = np.nan_to_num(X_train, nan=-1)
            X_test = np.nan_to_num(X_test, nan=-1)
            X_val = np.nan_to_num(X_val, nan=-1)

            es = EarlyStopping(monitor='val_loss', verbose=0, patience=150, restore_best_weights=True)

            input = Input(shape=(None, X_train.shape[-1]))
            lstm = Masking(mask_value=-1)(input)
            if self.args.model == 'GRU':
                lstm = GRU(self.args.n_neurons, dropout=0.5, 
                           recurrent_dropout=0.5, return_sequences=True, 
                           activation="sigmoid")(lstm)
            elif self.args.model == 'LSTM':
                lstm = LSTM(self.args.n_neurons, dropout=0.5, 
                            recurrent_dropout=0.5, return_sequences=True
                            )(lstm)
            else:
                exit('WRONG MODEL NAME')
            lstm = Dense(1, activation="sigmoid")(lstm)

            model = Model(input, lstm)
            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1000, callbacks=[es], verbose=0)
            best_epoch.append(np.argmin(history.history['val_loss']))

            model.save(f"results/{self.path}/{self.args.n_neurons}/weights/{i}.h5")

            predictions_raw = model.predict(X_test)[:, :, 0]
            predictions = np.round(predictions_raw)
            y_test = y_test[:, :, 0]
            for p_idx in range(len(predictions)):
                for v in range(self.n_max_visits):
                    if X_test[p_idx, v].max() != -1:
                        self.predictions_values_per_visit[v][i].append(predictions[p_idx, v])
                        self.predictions_raw_values_per_visit[v][i].append(predictions_raw[p_idx, v])
                        self.real_values_per_visit[v][i].append(y_test[p_idx, v])
                    else:
                        break
            print(f'Best epoch: train: {np.max(history.history["accuracy"])} valid: {np.max(history.history["val_accuracy"])}')
        print('Average epochs: ', np.mean(best_epoch), "+-", np.std(best_epoch))
        os.makedirs(f"results/{self.path}/{self.args.n_neurons}/", exist_ok=True)
        pickle.dump(self.predictions_values_per_visit,
                    open(f"results/{self.path}/{self.args.n_neurons}/predictions.pkl", "wb"))
        pickle.dump(self.predictions_raw_values_per_visit,
                    open(f"results/{self.path}/{self.args.n_neurons}/predictions_raw.pkl", "wb"))
        pickle.dump(self.real_values_per_visit,
                    open(f"results/{self.path}/{self.args.n_neurons}/real_values.pkl", "wb"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_features", type=int, default=-1)
    parser.add_argument("--scaler", type=str, default="minmax")
    parser.add_argument("--features_file", type=str, default="features")
    parser.add_argument("--labels_file", type=str, default="labels")
    parser.add_argument("--patients_file", type=str, default="patients.pkl")
    parser.add_argument("--data_path", type=str, default="data")
    # parser.add_argument("--indices_file", type=str, default="features/split_indices.csv")
    parser.add_argument("--model", type=str, default="GRU", help='choose one of [GRU, LSTM]')
    parser.add_argument("--n_splits", type=int, default=100)
    parser.add_argument("--endpoint", type=str, default='2', help="5 or 2 years ['2', '5']")
    parser.add_argument("--n_neurons", type=int, default=16)
    parser.add_argument("--gpu_id", type=str, default='1')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

    args.labels_file = f"{args.labels_file}_{args.endpoint}.pkl"
    if args.n_features == -1:
        args.features_file = f"{args.features_file}.pkl"
    else:
        args.features_file = f"{args.features_file}-{args.n_features}.pkl"

    os.makedirs(f'results/{args.model}', exist_ok=True)

    classifier = RNN(args)
    classifier.load_data(
        f"{args.data_path}/{args.features_file}",
        f"{args.data_path}/{args.labels_file}",
        f"{args.data_path}/{args.patients_file}"
    )
    classifier.train()


if __name__ == "__main__":
    main()
