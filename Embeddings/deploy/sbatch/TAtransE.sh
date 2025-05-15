#!/bin/bash
#SBATCH --Something
#ADD YOUR SLURM OPTIONS HERE

#Name of the output file must be - model_icuNcit_KgType_dim

uv venv .venv
# ================================== SIMPLE ==========================================
#CREATE THE FILES FOR GENRAL SIMPLE TEMPORAL PROCESS
uv run python3 -u Embeddings/embeddings/emb-TAtransE/data_conversion.py \
        --entity /datasets/ICU/Ncit_simple_time/entity2id.txt \
        --relation /datasets/ICU/Ncit_simple_time/relation2id.txt \
        --triples /datasets/ICU/Ncit_simple_time/PREFT.txt \
        --outdir /Embeddings/embeddings/resources/TA_TransE/data \
        --dataset MIMIC_TA

#CREATE THE FILES FOR TA_TRANSE - TEMPORAL INDEXING
uv run python3 -u Embeddings/embeddings/emb-TAtransE/preprocess_TA.py \
        --dataset MIMIC_TA \
        --datapath /Embeddings/embeddings/resources/TA_TransE/data

#RUN THE MODEL
uv run python3 -u Embeddings/embeddings/emb-TAtransE/runTAtransE.py \
        --dataset MIMIC \
        --datapath /Embeddings/embeddings/resources/TA_TransE/data \
        --target_entity_file /datasets/Targets.txt \
        --entity2id_file /datasets/ICU/Ncit_simple_time/entity2id.txt \
        --output_path /path_to_out_dir\
        --experiment TA-transE_icuNcit_simple_300

# ================================== SEMANTIC ==========================================

#CREATE THE FILES FOR GENRAL SEMANTIC TEMPORAL PROCESS
uv run python3 -u Embeddings/embeddings/emb-TAtransE/data_conversion.py \
        --entity /datasets/ICU/Ncit_semantic_time/entity2id.txt \
        --relation /datasets/ICU/Ncit_semantic_time/relation2id.txt \
        --triples /datasets/ICU/Ncit_semantic_time/MergedTrainRef.txt \
        --outdir /Embeddings/embeddings/resources/TA_TransE/data \
        --dataset MIMIC_Semantic_TA

#CREATE THE FILES FOR TA_TRANSE - TEMPORAL INDEXING
uv run python3 -u Embeddings/embeddings/emb-TAtransE/preprocess_TA.py \
        --dataset MIMIC_Semantic_TA \
        --datapath /Embeddings/embeddings/resources/TA_TransE/data

#RUN THE MODEL
uv run python3 -u Embeddings/embeddings/emb-TAtransE/runTAtransE.py \
        --dataset MIMIC_Semantic \
        --datapath /Embeddings/embeddings/resources/TA_TransE/data \
        --target_entity_file /datasets/Targets.txt \
        --entity2id_file /datasets/ICU/Ncit_semantic_time/entity2id.txt \
        --output_path /path_to_out_dir\
        --experiment TA-transE_icuNcit_semantic_300