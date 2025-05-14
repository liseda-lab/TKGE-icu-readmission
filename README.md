# **From Triples to Timelines: Temporal Knowledge Graph Embeddings for Patient-Centric ICU Readmission Prediction** #
_A Reproducible Analysis of Knowledge Graph Embeddings (KGE) and Temporal Knowledge Graph Embeddings (TKGE) for ICU Readmission Prediction_

## **🚀 Project Overview**
This repository contains the code for data processing, implementation, and analysis of our approach to predicting ICU readmissions using KGEs and TKGEs. Our project investigates how semantic enrichment and temporal representation impact predictive performance by generating multiple knowledge graphs with varying levels of semantic and temporal awareness. We then evaluate and compare static and temporal embedding methods, providing insights into the optimal configurations for clinical predictive tasks. The project has three main componenets: I) TKGs Construction, II) Embedding Contruction, III) Prediction.


## **📌 Key Features**
- Construction of four types of Knowledge Graphs (KGs): Simple, Temporal, Semantic, and Semantic-Temporal.
- Evaluation of multiple embedding strategies:
  - Static Embeddings: TransE, DistMult, ComplEx, RDF2Vec.
  - Temporal Embeddings: T-TransE, TA-TransE, T-ComplEx, TNT-ComplEx, TA-DistMult.
- Predictive models implementation (Logistic Regression, Random Forest, XGBoost).
- Comprehensive performance evaluation using accuracy and ROC AUC.
- Enabling analysis of embedding strategies across semantic and temporal axes.

## III) Prediction

Note: To run this portion of the analysis you need to have the MIMIC-III folder. If you opted to download the raw files to do the TKGs construcction you now need to download the MIMIC-III files. Otherwise proced.

As per the other septs please make shore you have uv installed and inside the directory sync the project

````python
    cd ReadmissionPrediction
    uv sync
````

Before the the prediction first generate the training data running:

````python
    #Process the files from MIMIC-III and generate the data folds
    uv run python dataSet_construction.py  <path_to_mimic_folder>
    uv run python data_preparation.py
````

To run the predictions you need to follow the following steps for each model/embedding pairing. If you are running using batch access the deploy folder for the files. Otherwise run as showned bellow. If needed the batch files can provide a reference of the pairings.

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