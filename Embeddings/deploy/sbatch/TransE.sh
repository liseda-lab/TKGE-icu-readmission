#!/bin/bash
#SBATCH --Something
#ADD YOUR SLURM OPTIONS HERE

#Structure of the script:
# 1. Script
# 2. Data Directory
# 3. Output Directory
# 4. Targetsfile Path
# 5. Name of the output file - model_icuNcit_KgType_dim


uv venv .venv
#Experiment for Embeddings with the semantic layer
uv run python Embeddings/embeddings/emb-transE/runTransE-OpenKE.py \
      /datasets/ICU/Ncit_semantic_notime/ \
      /path_to_out_dir \
      /datasets/Targets.txt \
      transE_icuNcit_semantic_300


#Experiment with no semantic layer
uv run python Embeddings/embeddings/emb-transE/runTransE-OpenKE.py \
      /datasets/ICU/Ncit_simple_notime/ \
      /path_to_out_dir \
      /datasets/Targets.txt \
      transE_icuNcit_simple_300