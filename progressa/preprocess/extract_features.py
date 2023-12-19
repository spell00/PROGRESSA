# file: new_extract_features.py
# Main author: Melissa Sanabria
# Comments and post-processing: Simon Pelletier & Louis Ohl

import pickle
import pandas as pd
import numpy as np

import os

import argparse

# Louis addition
from sklearn import preprocessing


all_categ_cols = ["sex", "NYHA", "hypertension", "metabolic_syndrome", "diabetes",
              "coronary artery disease", "previous_myocardial_infarction",
              "history of smoking", "history of atrial fibrillation",
              "COPD", "stroke_or_tia", "CABG", "PCI", "ARA", "ACE", "other_anti_HT",
              "beta_blockers", "calcium_blockers", "diuretics", "anti_lipid",
              "fibrates", "medication_diabetes", "anti_platelet", "anticoagulant",
              "biphosphonate", "medication_calcium", "medication_vitamin_d",
              "symp_angina", "symp_ex_dysp", "symp_rest_dysp", "symp_pre_synco",
              "symp_syncope", "symp_palpitations", "AO_valve_phenotype",
              "ao_regurgitation_grade", "mitral_regurgitation_grade",
              "tricuspid_regurgitation_grade", "grade_diastolic_dysfunction", "tertiles_delta_AO_Vpeak", "AS_severity"]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="2020_12_30_Databse_PROGRESSA.xlsx")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--features_output", type=str, default="features.pkl")
    parser.add_argument("--patients_output", type=str, default="patients.pkl")
    parser.add_argument("--labels_output", type=str, default="labels")
    parser.add_argument("--feature_names_output", type=str, default="feature_names.pkl")
    parser.add_argument("--endpoint", type=str, default='2', help="Final endpoint or 2 years ['2', 'final'] ")

    args = parser.parse_args()

    args.dataset = f"{args.data_path}/{args.dataset}"
    args.features_output = f"{args.data_path}/{args.features_output}"
    args.patients_output = f"{args.data_path}/{args.patients_output}"
    args.labels_output = f"{args.data_path}/{args.labels_output}_{args.endpoint}.pkl"
    args.feature_names_output = f"{args.data_path}/{args.feature_names_output}"

    assert os.path.exists(args.dataset), "Please provide an existing file for the original data"

    return args


def load_data(args):
    """
    Loads the PROGRESSA database and extract all sheets of data

    The sheets are then merged to create a single dataframe
    + The merge is a left join using Patients/visite IDs of the main sheet as key
    """
    main_DB = pd.read_excel(args.dataset, sheet_name = 'Main Database')
    MRI_DB = pd.read_excel(args.dataset, sheet_name = 'MRI Database')
    sphygmocor_DB = pd.read_excel(args.dataset, sheet_name = 'Sphygmocor Database')
    DXA_DB = pd.read_excel(args.dataset, sheet_name = 'DXA Database')
    clinical_outcomes = pd.read_excel(args.dataset, sheet_name = 'Clinical outcome')

    # There are initially two lines for the header, so we need to delete the first row
    # And use the second one as header

    main_DB_r = main_DB.copy()
    main_DB_r.columns = main_DB_r.iloc[0].values
    main_DB_r.drop(0, inplace=True)

    # ======================================

    # The main database is now in main_DB_r with the header correction
    main_DB_r = main_DB_r[main_DB_r["PSA ID"].isin(clinical_outcomes["PSA ID"])]

    # We now merge (left joins) all sheets of the excel file
    # As some sheets store the N/A values with "." or other smyobls, we correct those on the fly

    ## main and MRI
    concat_df = pd.merge(main_DB_r, MRI_DB, on=["PSA ID", "visite_id"], how="left", suffixes=["", "_y"])
    ## main, MRI and sphygmocor
    concat_df = pd.merge(concat_df, sphygmocor_DB, on=["PSA ID", "visite_id", "HR"], how="left")

    ## main, MRI, sphygmocor and DXA
    concat_df = pd.merge(concat_df, DXA_DB, on=["PSA ID", "visite_id"], how="left")

    concat_df.replace(".", "", inplace=True)

    duplicated_cols = [col for col in concat_df.columns if col.endswith("_x") or col.endswith("_y")]
    concat_df.drop(duplicated_cols, axis = 1, inplace = True)

    date_cols = [col for col in concat_df.columns if 'date' in col]
    concat_df.drop(date_cols, axis = 1, inplace = True)

    useless_cols = ["visite_id", "albumin", "CT_prog_ID"]
    concat_df.drop(useless_cols, axis=1, inplace=True)

    concat_df = concat_df.fillna(9999)

    return concat_df, clinical_outcomes, main_DB_r, MRI_DB, sphygmocor_DB, DXA_DB


def filter_data(concat_df):
    """
    Eliminates variables that have more than 5% of missing values per visit on all visits
    """
    ## Check the number of rows with missing values per visit. Then find the features that have missing values in all the visits
    n_max_visits =  concat_df[(concat_df["study_phase"] != "PO1y")]["visit_number"].values.max()

    na_dataset = concat_df.copy()
    na_dataset.replace(9999, np.nan, inplace=True)
    na_dataset.replace("", np.nan, inplace=True)

    na_dataset = na_dataset.isna()
    na_dataset["visit_number"] = concat_df["visit_number"]

    # Count the ratio of missing values per visit
    na_dataset = na_dataset.groupby(["visit_number"]).agg("mean")
    columns_to_delete = na_dataset.columns[(na_dataset>=0.05).sum(0)==len(na_dataset)]
    # ================================================

    concat_df.drop(columns_to_delete, axis = 1, inplace = True)

    return concat_df, n_max_visits


def preprocess_data(concat_df, n_max_visits, clinical_outcomes, endpoint, main_DB_r, MRI_DB, sphygmocor_DB, DXA_DB):
    global all_categ_cols

    patients = concat_df["PSA ID"].unique()

    # Locate the non-binary categorical variables
    non_binary_categ_cols = [x for x in all_categ_cols if len(concat_df[x].unique())>2]
    onehot_encoder = preprocessing.OneHotEncoder(categories="auto", drop="if_binary", sparse=False)
    onehot_encoded_variables = onehot_encoder.fit_transform(concat_df[non_binary_categ_cols])
    continuous_variables_df = concat_df.drop(non_binary_categ_cols + ["study_phase"], axis=1).replace("", np.nan).replace(9999, np.nan)

    # We remove patients that are in the study phase PO1y
    rows_to_keep = concat_df.index[concat_df.study_phase!="PO1y"]
    onehot_encoded_variables = onehot_encoded_variables[rows_to_keep]
    continuous_variables_df = continuous_variables_df.iloc[rows_to_keep]

    # We need to compute the delta of AO_mean_gradient, AO_Vpeak, AVA
    # So we extract the matching columns with PSA and visite_number
    initial_columns = ["AO_mean_gradient", "AO_Vpeak", "AVA"]
    delta_columns = [f"delta_{col}" for col in initial_columns]
    delta_df = concat_df.iloc[rows_to_keep][["PSA ID", "visit_number"]+initial_columns].replace("", np.nan).replace(9999, np.nan)
    # Compute delta per visit
    delta_df[delta_columns] = -delta_df.sort_values(by=["PSA ID", "visit_number"])[initial_columns].diff()
    delta_df["delta_AVA"] = -delta_df["delta_AVA"]
    # Assign values 0 to all samples in visit number 0
    delta_df.loc[(delta_df.visit_number==1), delta_columns] = 0


    onehot_encoded_variables_df = pd.DataFrame(onehot_encoded_variables, columns=onehot_encoder.get_feature_names_out())
    features_df = pd.concat([pd.merge(left=continuous_variables_df, right=delta_df[["PSA ID", "visit_number"]+delta_columns], how="left"), onehot_encoded_variables_df], axis=1)
    features = np.ones((len(features_df["PSA ID"].unique()), n_max_visits, features_df.shape[1]-2)) * np.nan
    patients_to_delete = []
    for i, patient in enumerate(sorted(features_df["PSA ID"].unique())):
        subdf = features_df[features_df["PSA ID"]==patient].drop(["PSA ID", "visit_number"], axis=1)
        if len(subdf)==1:
            patients_to_delete += [i]
        else:
            features[i][:len(subdf)] = subdf.to_numpy()

    features = np.delete(features, patients_to_delete, axis=0)
    print("Final shape", features.shape)

    # =============================

    labels = np.zeros((len(patients), n_max_visits))

    endpoint = int(endpoint)
    for p_id, patient in enumerate(sorted(patients)):
        # visits_per_patient = concat_df[(concat_df["PSA ID"] == patient)& (concat_df["study_phase"] != "PO1y")]["visit_number"]
        # AVR_or_cardiovascular_death = clinical_outcomes[(clinical_outcomes["PSA ID"] == patient)]["AVR_or_cardiovascular_death"].values[0]
        # if AVR_or_cardiovascular_death == 1:
        #     labels[p_id, max(min(len(visits_per_patient), 6) - 3, 0):] = 1
        #     # print("avr")
        # severities = main_DB_r[(main_DB_r["PSA ID"] == patient) & (main_DB_r["study_phase"] != "PO1y")]["AS_severity"].values
        # if(len(severities)) > 1:
        #     for idx in range(0, len(severities) - 2):
        #         if severities[idx + 2] > severities[idx]:
        #             labels[p_id, min(idx, 6):] = 1
        #             # print("sev")
        #             break
        visits_per_patient = concat_df[(concat_df["PSA ID"] == patient)& (concat_df["study_phase"] != "PO1y")]["visit_number"]
        AVR_or_cardiovascular_death = clinical_outcomes[(clinical_outcomes["PSA ID"] == patient)]["AVR_or_cardiovascular_death"].values[0]
        if AVR_or_cardiovascular_death == 1:
            # -1 because indices start at 0
            labels[p_id, max(len(visits_per_patient) - endpoint - 1, 0):] = 1
            # labels[p_id, max(min(len(visits_per_patient), 6) - endpoint - 1, 0):] = 1
            # print("avr")
        severities = main_DB_r[(main_DB_r["PSA ID"] == patient) & (main_DB_r["study_phase"] != "PO1y")]["AS_severity"].values
        if(len(severities)) > 1:
            for idx in range(0, len(severities) - endpoint):
                if severities[idx + endpoint] > severities[idx]:
                    # labels[p_id, min(idx, 6):] = 1
                    labels[p_id, idx:] = 1
                    # print("sev")
                    break
        # print(patient, labels[p_id])

    ## As for the features, remove labels for patients with only one visit
    labels = np.delete(labels, patients_to_delete, axis=0)
    patients = np.delete(patients, patients_to_delete, axis=0)

    # Finally, generate the names of the features
    #[2:] gets rid of PSA ID and visit_number
    feature_names = [col.replace("_", " ") for col in features_df.columns[2:]]

    print(feature_names)

    return patients, features, labels, feature_names


def export_data(args, patients, features, labels, feature_names):
    print(f"Exporting features to: {args.features_output}_{args.endpoint}")
    pickle.dump(features[:, :6], open(args.features_output, "wb"))
    print(f"Exporting labels to: {args.labels_output}")
    pickle.dump(labels[:, :6], open(args.labels_output, "wb"))
    print(f"Exporting patients to: {args.patients_output}")
    pickle.dump(patients, open(args.patients_output, "wb"))
    print(f"Exporting feature names to: {args.feature_names_output}")
    pickle.dump(feature_names, open(args.feature_names_output, "wb"))


def main():
    args = get_args()

    print("Loading the sheets from the file: ", args.dataset)
    concat_df, clinical_outcomes, main_DB_R, MRI_DB, sphygmocor_DB, DXA_DB = load_data(args)
    print("Obtained a concatenated csv of shape: ", concat_df.shape)

    print("Filtering out variables that present more than 5% missing values per visit on all visits")
    concat_df, n_max_visits = filter_data(concat_df)
    print("Remaining shape of dataframe is: ", concat_df.shape)

    print("Preprocessing data and creating labels")
    patients, features, labels, feature_names = preprocess_data(concat_df, n_max_visits, clinical_outcomes, args.endpoint, main_DB_R, MRI_DB, sphygmocor_DB, DXA_DB)
    print(f"Shape of the features is: {features.shape} / Shape of labels: {labels.shape}")
    print(f"Retrieved {len(feature_names)} different features")

    print("Exporting all files")
    export_data(args, patients, features, labels, feature_names)

    print("FINISHED")


if __name__ == "__main__":
    main()
