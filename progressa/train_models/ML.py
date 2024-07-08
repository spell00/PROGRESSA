import os
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import lightgbm
from xgboost import XGBClassifier


class Classifier:
    """
    Classifier class
    """
    def __init__(self, args):
        """
        Classifier constructor
        """
        self.args = args
        self.seed = 1
        self.indices_dict = {}

        if args.scaler == 'standard':
            self.scaler = StandardScaler()
        elif args.scaler == 'robust':
            self.scaler = RobustScaler()
        elif args.scaler == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            self.scaler = None

        if  'Logistic_Regression' in self.args.model:
            self.model = LogisticRegression
        elif 'naiveBayes' in self.args.model:
            self.model = GaussianNB
        elif 'lightgbm' in self.args.model:
            self.model = lightgbm.LGBMClassifier
        elif 'xgboost' in self.args.model:
            self.model = XGBClassifier
        self.get_splits()
        print(self.args.n_splits)
        self.path = f"{self.args.model}/{self.args.scaler}/{self.args.n_features}_features/n{self.args.n_splits}/endpoint{self.args.endpoint}/"
        os.makedirs(f"results/{self.path}/weights", exist_ok=True)

    def make_indices_dict(self, i):

        self.indices_dict = {
            'train': np.argwhere(self.all_indices[:, i] == 1).flatten(),
            'test': np.argwhere(self.all_indices[:, i] == 2).flatten(),
            'valid': np.argwhere(self.all_indices[:, i] == 3).flatten()
        }


    def load_data(self, features_file, labels_file, patients_file):
        """
        Data getter
        """
        self.features = pickle.load(open(features_file, "rb"))
        self.labels = pickle.load(open(labels_file, "rb"))
        self.patients = pickle.load(open(patients_file, "rb"))
        self.n_max_visits = self.features.shape[1]
        self.scaler.fit(self.features.reshape(-1, self.features.shape[-1]))

        self.predictions_values_per_visit = {}
        for v in range(self.n_max_visits):
            self.predictions_values_per_visit[v] = [[] for i in range(self.args.n_splits)]
        self.predictions_values_per_visit[-1] = [[] for i in range(self.args.n_splits)]

        self.real_values_per_visit = {}
        for v in range(self.n_max_visits):
            self.real_values_per_visit[v] = [[] for i in range(self.args.n_splits)]
        self.real_values_per_visit[-1] = [[] for i in range(self.args.n_splits)]

        self.predictions_raw_values_per_visit = {}
        for v in range(self.n_max_visits):
            self.predictions_raw_values_per_visit[v] = [[] for i in range(self.args.n_splits)]
        self.predictions_raw_values_per_visit[-1] = [[] for i in range(self.args.n_splits)]

    def get_splits(self):
        indices_file = f"{self.args.data_path}/split_indices.csv"
        self.all_indices = np.loadtxt(open(indices_file), delimiter=",")
