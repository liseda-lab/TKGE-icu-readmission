#!/bin/bash
#SBATCH --Something
#ADD YOUR SLURM OPTIONS HERE

#Name of the output file must be - model_icuNcit_KgType_dim

uv venv .venv
##########################   SEMANTIC   #############################
# Generate Semantic Files - Convert to TcomplEx format
uv run python3 Embeddings/embeddings/emb-TNTcomplEx/tkbc_data_conversion.py \
        --entity /datasets/ICU/Ncit_semantic_time/entity2id.txt \
        --relation /datasets/ICU/Ncit_semantic_time/relation2id.txt \
        --triples /datasets/ICU/Ncit_semantic_time/MergedTrainRef.txt

# Run the embedding model
uv run python3 Embeddings/embeddings/emb-TNTcomplEx/runTNTcomplEx.py \
        --dataset /Embeddings/embeddings/resources/tkbc/tkbc/data \
        --model TComplEx \
        --targets /datasets/Targets.txt \
        --entity2id /datasets/ICU/Ncit_semantic_time/entity2id.txt \
        --output /path_to_out_dir \
        --experiment TcomplEx_icuNcit_semantic_300


##########################   SIMPLE   #############################
# Generate Simple Files - Convert to TcomplEx format
uv run python3 Embeddings/embeddings/emb-TNTcomplEx/tkbc_data_conversion.py \
        --entity /datasets/ICU/Ncit_simple_time/entity2id.txt \
        --relation /datasets/ICU/Ncit_simple_time/relation2id.txt \
        --triples /datasets/ICU/Ncit_simple_time/PREFT.txt

# Run the embedding model
uv run python3 Embeddings/embeddings/emb-TNTcomplEx/runTNTcomplEx.py \
        --dataset /Embeddings/embeddings/resources/tkbc/tkbc/data \
        --model TComplEx \
        --targets /datasets/Targets.txt \
        --entity2id /datasets/ICU/Ncit_simple_time/entity2id.txt \
        --output /path_to_out_dir \
        --experiment TcomplEx_icuNcit_simple_300