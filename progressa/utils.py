import numpy as np

def get_data(set_features, set_labels, indices):
    new_features = []
    new_labels = []
    patient_visit = []
    for p_idx in range(set_features.shape[0]):
        for v in range(set_features.shape[1]):
            if np.nan_to_num(set_features[p_idx, v], nan=-1).max() != -1:
                new_features.append(set_features[p_idx, v])
                new_labels.append(set_labels[p_idx, v])
                patient_visit.append(np.array([indices[p_idx], v]))
            else:
                pass
    set_features = np.asarray(new_features)
    set_labels = np.asarray(new_labels)
    patient_visit = np.asarray(patient_visit)

    return set_features, set_labels, patient_visit

def get_data_2v(set_features, set_labels, indices):
    new_features = []
    new_labels = []
    patient_visit = []
    for p_idx in range(set_features.shape[0]):
        for v in range(1, set_features.shape[1]):
            if np.nan_to_num(set_features[p_idx, v], nan=-1).max() != -1:
                new_features.append(set_features[p_idx, v - 1: v + 1].flatten())
                new_labels.append(set_labels[p_idx, v])
                patient_visit.append(np.array([indices[p_idx], v]))

    set_features = np.asarray(new_features)
    set_labels = np.asarray(new_labels)
    patient_visit = np.asarray(patient_visit)
    one_hot_labels = np.zeros((len(set_labels), 2))
    one_hot_labels[:, 0] += set_labels
    one_hot_labels[set_labels == 0, 1] = 1

    return set_features, set_labels, patient_visit
