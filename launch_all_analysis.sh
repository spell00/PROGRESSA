# Description: Launch all models in the project
# Loop over all models
n_splits=100
n_neurons=32
for endpoint in 2 5
do
    for n_features in -1
    do
        for scaler in "minmax"
        do
            python3 progressa/analysis/analyse_results.py --scaler=$scaler --n_features=$n_features --n_splits=$n_splits --endpoint=$endpoint --n_neurons=$n_neurons
            python3 progressa/analysis/analyse_results_per_feature.py --scaler=$scaler --n_features=$n_features --n_splits=$n_splits --endpoint=$endpoint --n_neurons=$n_neurons
            python3 progressa/create_images/plot_rocs.py --scaler=$scaler --n_features=$n_features --n_splits=$n_splits --endpoint=$endpoint --cumul=0 --n_neurons=$n_neurons
            python3 progressa/create_images/plot_rocs.py --scaler=$scaler --n_features=$n_features --n_splits=$n_splits --endpoint=$endpoint --cumul=1 --n_neurons=$n_neurons
        done
    done
done
