## 🎯 Goal
_This repository provides a comprehensive and reproducible framework for ICU readmission prediction using our TKGS from the MIMIC-III data._
_These models will leverage demographics + embeddings to predict with simple ML models XGB, LR and RF_

Note: _To run this portion of the analysis you need to have a MIMIC-III folder. If you opted to download the raw files to do the TKGs construcction you now need to download the MIMIC-III files. Otherwise proced._

## **📊 Usage**
Please make garantee you have uv installed and inside the directory sync the project

````python
    cd ReadmissionPrediction
    uv sync
````
### Data
Generate the training data:
````python
    #Process the files from MIMIC-III and generate the data folds
    uv run python dataSet_construction.py  <path_to_mimic_folder>
    uv run python data_preparation.py
````

### Implemnetation
Before runing garantee you embeddings are structured as embedding_icuNcit_KG_300_embedding.txt.

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


Run the following steps for each model/embedding/KG-stratefy pairing:

1. Generate the features (Demographics + Embeddings for the folds) :
````python
    uv run python feature_engineering.py complEx_icuNcit_semantic_300_embeddings.txt
````
2. Run the ML models for the model/embedding pairing:
````python
    uv run python ml_prediction.py complEx_semantic LogisticRegression
    uv run python ml_prediction.py complEx_semantic XGBoost
    uv run python ml_prediction.py complEx_semantic RandomForest
````

For the baseline run:
1. Generate the features:
````python
    uv run python baseline_feature_engineering.py
````
2. Run the ML models for the model/embedding pairing:
````python
    uv run python ml_prediction.py no_embedding LogisticRegression
    uv run python ml_prediction.py no_embedding XGBoost
    uv run python ml_prediction.py no_embedding RandomForest
````

After runnning you can optionaly do an analysis:
````python
    uv run python avg_metrics.py
````