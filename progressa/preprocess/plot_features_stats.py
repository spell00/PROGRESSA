import matplotlib.pyplot as plt
import pickle
import numpy as np

def main():
    max_scores = {endpoint: pickle.load(open(f"number_samples_{endpoint}.pkl", "rb")) for endpoint in [2, 5]}
    total_positives = {endpoint: pickle.load(open(f"total_positives_{endpoint}.pkl", "rb")) for endpoint in [2, 5]}
    n_max_visits = 11
    plt.bar(np.arange(n_max_visits) - 0.2, total_positives[2] / max_scores[2], 0.4, label='2 years')
    plt.bar(np.arange(n_max_visits) + 0.2, total_positives[5] / max_scores[5], 0.4, label='5 years')
    plt.ylabel("Proportion of positive samples")
    plt.xlabel("Visit number")
    plt.title(f"Proportion of positive samples per visit")
    plt.legend()
    plt.savefig(f"proportions_positives.png")
    plt.close()

    plt.bar(range(n_max_visits), max_scores[2]) # 2 and 5 are the same
    plt.ylabel("Number of samples")
    plt.xlabel("Visit number")
    plt.title(f"Number of samples per visit")
    plt.savefig(f"number_samples.png")
    plt.close()

    plt.bar(np.arange(n_max_visits) - 0.2, total_positives[2], 0.4, label='2 years')
    plt.bar(np.arange(n_max_visits) + 0.2, total_positives[5], 0.4, label='5 years')
    plt.ylabel("Number of positive samples")
    plt.xlabel("Visit number")
    plt.title(f"Number of positive samples per visit")
    plt.legend()
    plt.savefig(f"Number_positives.png")
    plt.close()


if __name__ == "__main__":
    main()