# Description: Launch all models in the project
# Loop over all models
for endpoint in 2 5
do
    for n_features in 22
    do
        for scaler in "minmax"
        do
            for model in "naiveBayes" "Logistic_Regression" "lightgbm" "xgboost"
            do
                python3 progressa/train_models/sklearn_models.py --model=$model --scaler=$scaler --n_features=$n_features --n_splits=100 --endpoint=$endpoint
                python3 progressa/train_models/sklearn_models_cumul.py --model=$model --scaler=$scaler --n_features=$n_features --n_splits=100 --endpoint=$endpoint
            done
        done
    done
done

for n_neurons in 16
do
    for endpoint in 2 5
    do
        for n_features in 22
        do
            for scaler in "minmax"
            do
                for model in "GRU" "LSTM"
                do
                    python3 progressa/train_models/RNN.py --model=$model --scaler=$scaler --n_features=$n_features --n_splits=100 --endpoint=$endpoint --gpu_id=-1 --n_neurons=$n_neurons
                done
            done
        done
    done
done
