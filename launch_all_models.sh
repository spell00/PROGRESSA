# Description: Launch all models in the project
# Loop over all models
for endpoint in 2 5
do
    for n_features in -1
    do
        for scaler in "robust"
        do
            for model in "naiveBayes" "Logistic_Regression" "lightgbm" "xgboost"
            do
                python3 progressa/train_models/sklearn_models.py --model=$model --scaler=$scaler --n_features=$n_features --n_splits=100 --endpoint=$endpoint
                python3 progressa/train_models/sklearn_models_cumul.py --model=$model --scaler=$scaler --n_features=$n_features --n_splits=100 --endpoint=$endpoint
            done
        done
    done
done

# for n_neurons in 16 8 4 2
# do
#     for endpoint in 2 5
#     do
#         for n_features in -1
#         do
#             for scaler in "standard" "minmax"
#             do
#                 for model in "GRU"
#                 do
#                     python3 progressa/train_models/RNN.py --model=$model --scaler=$scaler --n_features=$n_features --n_splits=100 --endpoint=$endpoint --gpu_id=-1 --n_neurons=$n_neurons
#                 done
#             done
#         done
#     done
# done
# 