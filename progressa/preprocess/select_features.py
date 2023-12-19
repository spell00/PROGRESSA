import pickle
import numpy as np
import argparse

# features = pickle.load(open("../features.pkl", "rb"))
def select_features(args):
    features = pickle.load(open(f"{args.data_path}/features_{args.endpoint}.pkl", "rb"))
    feature_names = pickle.load(open(f"{args.data_path}/feature_names.pkl", "rb"))

    features_to_select = [
        "tertiles_delta_AO_Vpeak1.0",
        "lateral_s_wave",
        "AVA_index",
        "aorta_ascending",
        "LV_posterior_wall_diastole",
        "ttopv",
        "interventricular_septum_diastole",
        "AS_severity2",
        "moderate_AS_progression_Echo",
        "AS_severity1",
        "cholesterol_total",
        "AVA",
        "AO_Vpeak",
        "tertiles_delta_AO_Vpeak3.0",
        "septal_a_wave",
        "AO_vti",
        "proteines",
        "tertiles_delta_AO_Vpeak2.0",
        "echo_heart_rate",
        "delta_AVA",
        "mean_glomerular_volume",
        "LV_mass"
    ]

    new_features = np.zeros(features[:, :, :len(features_to_select)].shape)
    count = 0
    for idx_orig, f_n_orig in enumerate(feature_names):
        if f_n_orig in features_to_select:
            new_features[:, :, count] = features[:, :, idx_orig]
            count += 1

    # pickle.dump(new_features, open("features-"+str(len(features_to_select))+".pkl", "wb"))
    pickle.dump(new_features, open(f"{args.data_path}/features-{len(features_to_select)}_{args.endpoint}.pkl", "wb"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/")
    parser.add_argument("--endpoint", type=int)

    args = parser.parse_args()

    select_features(args)


if __name__ == "__main__":
    main()

