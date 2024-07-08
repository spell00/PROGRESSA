# Description: Launch all models in the project
# Loop over all models
n_splits=100
n_neurons=16
results_path="results"
for endpoint in 2 5
do
    for n_features in 22 -1_22
    do
        for scaler in "minmax"
        do
            python3 progressa/analysis/analyse_results.py --scaler=$scaler --n_features=$n_features --n_splits=$n_splits --endpoint=$endpoint --n_neurons=$n_neurons --results_path=$results_path
            python3 progressa/analysis/analyse_results_per_feature.py --scaler=$scaler --n_features=$n_features --n_splits=$n_splits --endpoint=$endpoint --n_neurons=$n_neurons --results_path=$results_path
            python3 progressa/analysis/get_shap.py --scaler=$scaler --n_features=$n_features --n_splits=$n_splits --endpoint=$endpoint --cumul=0 --n_neurons=$n_neurons --results_path=$results_path
            python3 progressa/create_images/plot_rocs.py --scaler=$scaler --n_features=$n_features --n_splits=$n_splits --endpoint=$endpoint --cumul=0 --n_neurons=$n_neurons --results_path=$results_path --all_models=0
            python3 progressa/create_images/plot_rocs.py --scaler=$scaler --n_features=$n_features --n_splits=$n_splits --endpoint=$endpoint --cumul=1 --n_neurons=$n_neurons --results_path=$results_path --all_models=0
            python3 progressa/create_images/plot_rocs.py --scaler=$scaler --n_features=$n_features --n_splits=$n_splits --endpoint=$endpoint --cumul=0 --n_neurons=$n_neurons --results_path=$results_path --all_models=1
            python3 progressa/create_images/plot_rocs.py --scaler=$scaler --n_features=$n_features --n_splits=$n_splits --endpoint=$endpoint --cumul=1 --n_neurons=$n_neurons --results_path=$results_path --all_models=1
        done
    done
done
