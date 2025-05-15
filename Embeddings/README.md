## 🎯 Goal
_This folder has the code and dependencies to run embedding models on our mimic-iii ICU readmission data set_

For this work we used several resources from other authors.

| Repo | Embeddings | URL  
|-----------|-----------|-----------|
| TKBC | T-complEx, TNT-complEx | [TKBC](https://github.com/facebookresearch/tkbc.git) |
| OpenKE | TransE, complEx, distMult | [OpenKE](https://github.com/thunlp/OpenKE.git) |
| INK-USC | T-TransE, TA-TransE, TA-distMult | [INK-USC](https://github.com/INK-USC/RE-Net.git) |


_For details about the embedding methods please refer to the corresponding papers_

- TransE: [Translating Embeddings for Modeling Multi-relational Data](https://proceedings.neurips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html).
- T-TransE: [Deriving Validity Time in Knowledge Graph](https://dl.acm.org/doi/abs/10.1145/3184558.3191639).
- TA-TransE, TA-DistMult: [Learning Sequence Encoders for Temporal Knowledge Graph Completion](https://arxiv.org/abs/1809.03202).
- distMult: [Embedding Entities and Relations for Learning and Inference in Knowledge Bases](https://arxiv.org/abs/1412.6575).
- complEx: [Complex Embeddings for Simple Link Prediction](https://proceedings.mlr.press/v48/trouillon16.html?ref=https://githubhelp.com).
- T-complEx, TNT-complEx: [Tensor Decompositions for temporal knowledge base completion](https://arxiv.org/abs/2004.04926).

## **📊 Usage**

1. Navigate to the resources folder and download or clone the three repositories needed for the embedding methods. These must be placed on the resources folder.
````python
    cd resources
    git clone https://github.com/facebookresearch/tkbc.git
    git clone https://github.com/thunlp/OpenKE.git
    git clone https://github.com/INK-USC/RE-Net.git
````

2. Navigate to the deploy folder and run the bacth files for each model. *The same script has the code for simple and semantic versions!
````python
    cd deploy
    sbatch deploy/ComplEx.sh
````

3. In case runing with batch is uncesserary, run the scripts in the files directly on the command line.

## **📌 Key Features**

_The embeddings generated will be used on the ML prediction framework. In case you desire just the embedding files, please reach out._