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

## **📊 Usage**

To use/reproduce this study, please consult the README.md files of each particular step. 
Run the resources in the following order:
1. TKGs Construction
````python
    cd TKG construction
````   
2. Embedding Contruction
````python
    cd Embedings
````   
3. Readmission Prediction
````python
    cd ReadmissionPrediction
````   