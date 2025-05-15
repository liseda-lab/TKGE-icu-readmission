#!/bin/bash
#SBATCH --Something
#ADD YOUR SLURM OPTIONS HERE

#Name of the output file must be - model_icuNcit_KgType_dim


uv venv .venv
# ================================== SIMPLE ==========================================
#CREATE THE FILES FOR GENRAL SIMPLE TEMPORAL PROCESS
uv run python3 /home/rcarvalho/Embeddings/embeddings/emb-TtransE/data_conversion.py \
        --entity /home/rcarvalho/datasets/ICU/Ncit_simple_time/entity2id.txt \
        --relation /home/rcarvalho/datasets/ICU/Ncit_simple_time/relation2id.txt \
        --triples /home/rcarvalho/datasets/ICU/Ncit_simple_time/PREFT.txt \
        --outdir /home/rcarvalho/Embeddings/embeddings/resources/TA_TransE/data \
        --dataset MIMIC_TTransE

#CREATE THE FILES FOR T_TRANSE - TEMPORAL INDEXING
uv run python3 Embeddings/embeddings/emb-TtransE/preprocess_TTransE.py \
        --dataset MIMIC_TTransE \
        --datapath /Embeddings/embeddings/resources/TA_TransE/data

#RUN THE MODEL
uv run python3 -u Embeddings/embeddings/emb-TtransE/runTTransE.py \
        --dataset MIMIC \
        --datapath //Embeddings/embeddings/resources/TA_TransE/data \
        --entity /datasets/ICU/Ncit_simple_time/entity2id.txt \
        --target_entity_file /datasets/Targets.txt \
        --output_path /path_to_out_dir\
        --experiment TTransE_icuNcit_simple_300


# ================================== SEMANTIC ==========================================
#CREATE THE FILES FOR GENRAL SEMANTIC TEMPORAL PROCESS
uv run python3 Embeddings/embeddings/emb-TtransE/data_conversion.py \
        --entity /datasets/ICU/Ncit_semantic_time/entity2id.txt \
        --relation /datasets/ICU/Ncit_semantic_time/relation2id.txt \
        --triples /datasets/ICU/Ncit_semantic_time/MergedTrainRef.txt \
        --outdir /Embeddings/embeddings/resources/TA_TransE/data \
        --dataset MIMIC_Semantic_TTransE

# #CREATE THE FILES FOR T_TRANSE - TEMPORAL INDEXING
uv run python3 Embeddings/embeddings/emb-TtransE/preprocess_TTransE.py \
        --dataset MIMIC_Semantic_TTransE \
        --datapath /Embeddings/embeddings/resources/TA_TransE/data

#RUN THE MODEL
uv run python3 -u Embeddings/embeddings/emb-TtransE/runTTransE.py \
        --dataset MIMIC_Semantic \
        --datapath /Embeddings/embeddings/resources/TA_TransE/data \
        --entity /datasets/ICU/Ncit_semantic_time/entity2id.txt \
        --target_entity_file /datasets/Targets.txt \
        --output_path /path_to_out_dir\
        --experiment TTransE_icuNcit_semantic_300