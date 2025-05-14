

Note: To run this portion of the analysis you need to have a MIMIC-III folder. If you opted to download the raw files to do the TKGs construcction you now need to download the MIMIC-III files. Otherwise proced.

As per the other septs please make shore you have uv installed and inside the directory sync the project

````python
    cd ReadmissionPrediction
    uv sync
````
## Data
Before the the prediction first generate the training data running:

````python
    #Process the files from MIMIC-III and generate the data folds
    uv run python dataSet_construction.py  <path_to_mimic_folder>
    uv run python data_preparation.py
````

## Implemnetation
Before runing garantee you embeddings are struturede as embedding_icuNcit_KG_300_embedding.txt.

Overview of the options:

| Embeddings | KG strategy | Model |
|-----------|-----------|-----------|
| complEx | simple | LogisticRegression |
| TcomplEx | semantic | XGBoost |
| TNTcomplEx | -- | RandomForest |
| TransE| -- | -- |
| TtransE | --| -- |
| TAtransE | --| -- |
| distmult | -- | -- |
| TAdistmult | --| -- |


To run the predictions you need to do the following steps for each model/embedding/KG-stratefy pairing:

1. Genrerate the features (Demographics + Embeddings for the folds) :
````python
    uv run python feature_engineering.py complEx_icuNcit_semantic_300_embeddings.txt
````
2. Run the ML models for the  model/embedding pairing:
````python
    uv run python ml_prediction.py complEx_semantic LogisticRegression
    uv run python ml_prediction.py complEx_semantic XGBoost
    uv run python ml_prediction.py complEx_semantic RandomForest
````
3. Adjust the avg_metrics.script to run for the models you desire and run:
````python
    uv run python avg_metrics.py
````